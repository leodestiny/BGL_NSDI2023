import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import sklearn.linear_model as lm
import sklearn.metrics as skm

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pyinstrument import Profiler

def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    return batch_inputs

class NegativeSampler(object):
    def __init__(self, g, neg_nseeds):
        self.neg_nseeds = neg_nseeds

    def __call__(self, num_samples):
        # select local neg nodes as seeds
        return self.neg_nseeds[th.randint(self.neg_nseeds.shape[0], (num_samples,))]

class NeighborSampler(object):
    def __init__(self, g, fanouts, neg_nseeds, sample_neighbors, num_negs, remove_edge):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.neg_sampler = NegativeSampler(g, neg_nseeds)
        self.num_negs = num_negs
        self.remove_edge = remove_edge

    def sample_blocks(self, seed_edges):
        #print(len(seed_edges))
        start = time.time()
        n_edges = len(seed_edges)
        seed_edges = th.LongTensor(np.asarray(seed_edges))
        heads, tails = self.g.find_edges(seed_edges)

        neg_tails = self.neg_sampler(self.num_negs * n_edges)
        neg_heads = heads.view(-1, 1).expand(n_edges, self.num_negs).flatten()

        # Maintain the correspondence between heads, tails and negative tails as two
        # graphs.
        # pos_graph contains the correspondence between each head and its positive tail.
        # neg_graph contains the correspondence between each head and its negative tails.
        # Both pos_graph and neg_graph are first constructed with the same node space as
        # the original graph.  Then they are compacted together with dgl.compact_graphs.
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        seeds = pos_graph.ndata[dgl.NID]
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            if self.remove_edge:
                # Remove all edges between heads and tails, as well as heads and neg_tails.
                _, _, edge_ids = frontier.edge_ids(
                    th.cat([heads, tails, neg_heads, neg_tails]),
                    th.cat([tails, heads, neg_tails, neg_heads]),
                    return_uv=True)
                frontier = dgl.remove_edges(frontier, edge_ids)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)

            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        #print("really sampling time: {:.4f} s".format(time.time() - start))

        start = time.time()
        input_nodes = blocks[0].srcdata[dgl.NID]
        blocks[0].srcdata['features'] = load_subtensor(self.g, input_nodes, 'cpu')
        # Pre-generate CSR format that it can be used in training directly
        #print("really get feature time {:.4f} s".format(time.time() - start))
        return pos_graph, neg_graph, blocks

class PosNeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def compute_acc(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], train_nid,
                              dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        #dataset=train_nid.numpy(),
        dataset=np.arange(len(train_nid)),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)

    # Define model and optimizer
    model = DistSAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_size = th.sum(g.ndata['train_mask'][0:g.number_of_nodes()])

    # Training loop
    iter_tput = []
    iter_time = []
    iter_sampling_time = []
    iter_feature_time = []
    iter_compute_time = []
    profiler = Profiler()
    if args.close_profiler == False:
        profiler.start()
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        step_start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for step, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            batch_inputs = blocks[0].srcdata['features']

            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            optimizer.step()
            update_time += time.time() - compute_end
            pos_edges = pos_graph.number_of_edges()
            if g.rank() == 0:
                iter_sampling_time.append(tic_step - step_start)
                iter_feature_time.append(start - tic_step)
                iter_compute_time.append(time.time() - start)     
            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(pos_edges/ step_t)
            iter_time.append(time.time() - step_start)
            if step % args.log_every == 0:
                speed = pos_edges * args.global_rank / np.mean(iter_time[-5*args.log_every:])
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s'.format(
                    g.rank(), epoch, step, loss.item(),speed, gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
                print('Avg sampling time {:.4f}s | Feature time {:.4f}s | Compute time {:.4f}s.'.format(
                        np.mean(iter_sampling_time[-5*args.log_every:]), np.mean(iter_feature_time[-5*args.log_every:]), np.mean(iter_compute_time[-5*args.log_every:])))
            start = time.time()
            step_start = time.time()

        toc = time.time()
        print('Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            g.rank(), toc - tic, sample_time, forward_time, backward_time, update_time, num_seeds, num_inputs))
        epoch += 1

    if args.close_profiler == False:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print('rank:', g.rank())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata['val_mask'], pb, force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata['test_mask'], pb, force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print('part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})'.format(
        g.rank(), len(train_nid), len(np.intersect1d(train_nid.numpy(), local_nid)),
        len(val_nid), len(np.intersect1d(val_nid.numpy(), local_nid)),
        len(test_nid), len(np.intersect1d(test_nid.numpy(), local_nid))))
    if args.num_gpus == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:'+str(g.rank() % args.num_gpus))
    labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print('#labels:', n_classes)

    # Pack data
    in_feats = g.ndata['features'].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help="the number of GPU device. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--fan_out', type=str, default='5,10,15')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_negs', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--close_profiler', action='store_true', help='Close pyinstrument profiler')
    args = parser.parse_args()
    assert args.num_workers == int(os.environ.get('DGL_NUM_SAMPLER')), \
    'The num_workers should be the same value with DGL_NUM_SAMPLER.'
    assert args.num_servers == int(os.environ.get('DGL_NUM_SERVER')), \
    'The num_servers should be the same value with DGL_NUM_SERVER.'

    print(args)
    main(args)