import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm

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

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, device):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device

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

        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]
        batch_inputs, batch_labels = load_subtensor(self.g, seeds, input_nodes, "cpu")
        blocks[0].srcdata['features'] = batch_inputs
        blocks[-1].dstdata['labels'] = batch_labels
        return blocks

class GAT(nn.Module):
    def __init__(self,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            num_heads,
            activation,
            dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2,allow_zero_in_degree=True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden, num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2,allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_classes, num_heads=num_heads, feat_drop=0., attn_drop=0., activation=None, negative_slope=0.2,allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, x, batch_size, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.n_hidden * num_heads if l != len(self.layers) - 1 else self.n_classes)
            else:
                y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    th.arange(g.number_of_nodes()),
                    sampler,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1) 
                else:
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                y[output_nodes] = h.cpu()

            x = y
        return 


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, 4, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors, device)

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)

    # Define model and optimizer
    num_heads = 4
    model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
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
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']
            batch_labels = batch_labels.long()

            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            blocks = [block.to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            optimizer.step()
            loss.item()
            update_time += time.time() - compute_end
            if g.rank() == 0:
                iter_sampling_time.append(tic_step - step_start)
                iter_feature_time.append(start - tic_step)
                iter_compute_time.append(time.time() - start)                
            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_time.append(time.time() - step_start)
            iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
            if step % args.log_every == 0 and g.rank() == 0:
                acc = compute_acc(batch_pred, batch_labels)
                speed = len(blocks[-1].dstdata[dgl.NID]) * args.global_rank / np.mean(iter_time[-5*args.log_every:])
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s'.format(
                    g.rank(), epoch, step, loss.item(), acc.item(), speed, gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
                print('Avg sampling time {:.4f}s | Feature time {:.4f}s | Compute time {:.4f}s.'.format(
                        np.mean(iter_sampling_time[-5*args.log_every:]), np.mean(iter_feature_time[-5*args.log_every:]), np.mean(iter_compute_time[-5*args.log_every:])))
            start = time.time()
            step_start = time.time()
        toc = time.time()
        print('Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            g.rank(), toc - tic, sample_time, forward_time, backward_time, update_time, num_seeds, num_inputs))
        epoch += 1


        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(model.module, g, g.ndata['features'],
                                         g.ndata['labels'], val_nid, test_nid, args.batch_size_eval, device)
            print('Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}'.format(g.rank(), val_acc, test_acc,
                                                                                  time.time() - start))
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
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_rank', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--close_profiler', action='store_true', help='Close pyinstrument profiler')
    args = parser.parse_args()
    assert args.num_workers == int(os.environ.get('DGL_NUM_SAMPLER')), \
    'The num_workers should be the same value with DGL_NUM_SAMPLER.'
    assert args.num_servers == int(os.environ.get('DGL_NUM_SERVER')), \
    'The num_servers should be the same value with DGL_NUM_SERVER.'

    print(args)
    main(args)