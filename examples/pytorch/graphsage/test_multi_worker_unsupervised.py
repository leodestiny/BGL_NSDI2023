import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import SAGEConv
import time
import math
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import sklearn.linear_model as lm
import sklearn.metrics as skm

import traceback
import queue
import os
import sys
import threading
from torch.nn.parallel import DistributedDataParallel

from load_graph import load_reddit, load_ogb, inductive_split
from dgl.distributed.shared_mem_utils import _to_shared_mem as tensor_to_shared_mem
from dgl.distributed import start_caching_process, start_training_recv_thread, get_sample_result, start_receiving_children_thread, dispatch_shared_memory_task, wait_receiving_remove_shared_memory_thread
from dgl.distributed import dispatch_unsuper_shared_memory_task, set_training_dgl_kernel_stream
from dgl.heterograph import DGLBlock
from dgl import backend

class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst

class DistNegativeSampler(object):
    def __init__(self, g, neg_nseeds):
        self.neg_nseeds = neg_nseeds

    def __call__(self, num_samples):
        # select local neg nodes as seeds
        return self.neg_nseeds[th.randint(self.neg_nseeds.shape[0], (num_samples,))]

class DistNeighborSampler(object):
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

        #start = time.time()
        #input_nodes = blocks[0].srcdata[dgl.NID]
        #blocks[0].srcdata['features'] = load_subtensor(self.g, input_nodes, 'cpu')
        # Pre-generate CSR format that it can be used in training directly
        #print("really get feature time {:.4f} s".format(time.time() - start))
        return pos_graph, neg_graph, blocks


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
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

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
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
                block = blocks[0].to(device)

                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def cal_mrr(pos_score, neg_score, num_negs=1):
    aff = pos_score.reshape([len(pos_score), 1])
    neg_aff = neg_score.reshape([len(pos_score), num_negs])
    score = th.cat([aff, neg_aff], axis=-1)
    size = score.shape[-1]
    _, indices_of_ranks = th.topk(score, k=size)
    _, ranks = th.topk(-indices_of_ranks, k=size)
    ranks = (ranks + 1)[:, 0]
    mrr = th.mean(1.0 / ranks.float())
    return mrr

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph, num_negs):
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
        mrr = cal_mrr(pos_score, neg_score, num_negs)
        return loss, mrr

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

def evaluate(model, g, inputs, labels, train_nids, val_nids, test_nids, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, inputs, batch_size, device)
        # multi gpu
        else:
            pred = model.module.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)


def read_nbytes(fd, n):
    ret = bytes()
    while n != 0:
        t = fd.read(n)
        n -= len(t)
        ret += t
    return ret

# create fifo for each process and each worker
def create_fifo(worker_num):
    for i in range(worker_num):
        recv_fifo_path = "/tmp/recv_fifo_w{:d}".format(i)
        cache_fifo_path = "/tmp/cache_fifo_w{:d}".format(i)
        train_fifo_path = "/tmp/train_fifo_w{:d}".format(i)
        if not os.path.exists(recv_fifo_path):
            os.mkfifo(recv_fifo_path)
        if not os.path.exists(cache_fifo_path):
            os.mkfifo(cache_fifo_path)
        if not os.path.exists(train_fifo_path):
            os.mkfifo(train_fifo_path)

def caching_process(args, device, data):
    in_feats, n_classes, train_g, val_g, test_g = data
    print("")
    start_caching_process(args.worker_num, args.number_of_nodes, args.cache_size_per_gpu, args.feature_dim, args.max_inputs_length, train_g.ndata['features'], args.samples_per_worker)

def training_process(args, device, data, worker_id):

    if args.n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = args.n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=worker_id)
    th.cuda.set_device(worker_id)
    in_feats, n_classes, train_g, val_g, test_g = data
    start_training_recv_thread(worker_id, args.max_inputs_length, args.feature_dim, args.layers, args.batch_size, th.zeros((0,)), args.samples_per_worker)

    labels = train_g.ndata['labels']

    # 0 to set no write buffer
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id),"wb", 0)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(worker_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[worker_id], output_device=worker_id)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(worker_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iter_time = []
    iter_sample = []
    iter_compute = []
    iter_tput = []
    epoch = 0

    cuda_stream = th.cuda.Stream()
    #with th.cuda.profiler.profile():
    #    with th.autograd.profiler.emit_nvtx():
    set_training_dgl_kernel_stream(worker_id, cuda_stream.cuda_stream)
    with th.cuda.stream(cuda_stream):
        for step in range(args.samples_per_worker):
            #print("in step {:d}".format(step))
            tic_step = time.time()

            sample_start = time.time()
            #blocks, batch_inputs, batch_labels = get_sample_result(args.layers, use_label=True)
            blocks, pos_graph, neg_graph, batch_inputs = get_sample_result(args.layers, use_label=False)
            sample_end = time.time()
            #print("convert time : {:.4f}s get ret_list: {:.4f}, block construct {:.4f}".format(time.time() - convert_start, convert_start - sample_start, block_end - block_start), batch_labels[0])
            #print(batch_inputs.shape)

            compute_start = time.time()
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss, mrr = loss_fcn(batch_pred, pos_graph, neg_graph, args.num_negs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            compute_end = time.time()
            pos_edges = pos_graph.number_of_edges()
            neg_edges = neg_graph.number_of_edges()

            batch_time = time.time() - tic_step
            #if batch_time < 0.1:
            iter_time.append(batch_time)
            iter_sample.append(sample_end-sample_start)
            iter_compute.append(compute_end-compute_start)
            iter_tput.append((pos_edges + neg_edges) * args.worker_num / batch_time)
            if step % args.log_every == 0 and worker_id == 0:
                print('AVERAGE Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), (pos_edges + neg_edges) * args.worker_num / np.mean(iter_time[3:]),  np.mean(iter_time[3:]), np.mean(iter_sample[3:]), np.mean(iter_compute[3:])))
                print('CURRENT Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), iter_tput[-1], iter_time[-1], sample_end-sample_start, compute_end-compute_start))
            # notify receving process this sample is finished 
            train_fifo.write(worker_id.to_bytes(4, sys.byteorder))
            train_fifo.write(step.to_bytes(4,sys.byteorder))
	  
    if args.n_gpus > 1:
        th.distributed.barrier()
    return


#### Entry point
def receiving_process(args, device, data, worker_id):

    max_moving_thread = 5

    start_receiving_children_thread(worker_id, args.layers, args.samples_per_worker, max_moving_thread)

    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    # sampler = NeighborSampler(train_g, [int(fanout) for fanout in args.fan_out.split(',')], train_nid,
    #                           dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)
    # dataloader = dgl.dataloading.DataLoader(
    #     dataset=train_nid,
    #     batch_size=args.batch_size,
    #     collate_fn=sampler.sample_blocks,
    #     shuffle=False,
    #     drop_last=False)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    # Unsupervised dataloader
    n_edges = train_g.number_of_edges()
    train_seeds = np.arange(n_edges)
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g, train_seeds, sampler, #exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        #reverse_eids=th.cat([
        #    th.arange(n_edges // 2, n_edges),
        #    th.arange(0, n_edges // 2)]),
        negative_sampler=NegativeSampler(train_g, args.num_negs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

#    max_pending_samples = 10
#    sampling_queue = queue.Queue(max_pending_samples)
#
#    # 0 to set no write buffer
#    recv_fifo = open("/tmp/recv_fifo_w{:d}".format(worker_id),"wb", 0)
#
#
#    # training process uses train_fifo to notify receiving process can release the shared memory
#    # receiving process reads info from train_fifo
#    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id), "rb", 0)
#
#
#    remove_shared_mem_thread = threading.Thread(target=receiving_process_remove_shared_mem_thread, args=(args, train_fifo, sampling_queue))
#    remove_shared_mem_thread.start()
#

    tmp_queue = queue.Queue()

    # Training loop
    avg = 0
    step = -1
    for epoch in range(args.num_epochs):
        if step == args.samples_per_worker:
            break
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        dataloader_start = time.time()
        #for pos_graph, neg_graph, blocks in dataloader:
        for input_nodes, pos_graph, neg_graph, blocks in dataloader:
            step += 1
            if step == args.samples_per_worker:
                break
            print("datalaoder takes {:.4f}s".format(time.time() - dataloader_start))
            
            #input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            ##move_blocks_to_shared_memory(blocks, input_nodes, seeds, args.num_layers)
            #to_shared_mem_start = time.time()
            #input_nodes = tensor_to_shared_mem(input_nodes, "w{:d}s{:d}_inputs".format(worker_id, step))
            #seeds = tensor_to_shared_mem(seeds, "w{:d}s{:d}_seeds".format(worker_id, step))
            #print("tensor to shared memory takes {:.4f}s".format(time.time() - to_shared_mem_start))
            #

            ## copy blocks into shared memory
            #for i in range(args.num_layers):
            #    to_shared_mem_start = time.time()
            #    # we create all formats for each block 
            #    # the graph name of each block is w0s0_b0, which means 'worker 0, step 0, block 0'
            #    blocks[i] = blocks[i].shared_memory("w{:d}s{:d}_b{:d}".format(worker_id, step, i), formats=("csr","csc"))
            #    to_shared_mem_end = time.time()
            #    print("block put to shared memory takes {:.4f}s".format(to_shared_mem_end-to_shared_mem_start))

            #
            #print("generate step {:d} with inputs shape {:d}".format(step, input_nodes.shape[0]))
            tmp_queue.put((worker_id, step, blocks, input_nodes, seeds, pos_graph, neg_graph))
            dataloader_start = time.time()


    for i in range(args.samples_per_worker):
            worker_id, step, blocks, input_nodes, seeds, pos_graph, neg_graph = tmp_queue.get()
            dispatch_unsuper_shared_memory_task(worker_id, step, blocks,  pos_graph, neg_graph,input_nodes, seeds)

            #sampling_queue.put((worker_id,step, blocks, input_nodes, seeds))

            # write info to recv_fifo
            # four parameter: worker_id, step, length of input_nodes, length of seeds
            # each parameter use 4 bytes (as int) with system byteorder, little endian in linux
            #recv_fifo.write(worker_id.to_bytes(4,sys.byteorder))
            #recv_fifo.write(step.to_bytes(4,sys.byteorder))
            #recv_fifo.write(input_nodes.shape[0].to_bytes(4,sys.byteorder))
            #recv_fifo.write(seeds.shape[0].to_bytes(4,sys.byteorder))
#            print("successfully put step {:d} in sampling queue size {:d}".format(step,input_nodes.shape[0]))
    wait_receiving_remove_shared_memory_thread()

            
    #remove_shared_mem_thread.join()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str,default='0,1,2,3,4,5,6,7',
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogb-product')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=1)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_negs', type=int, default=1)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
    argparser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    args.n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogb-product':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')
    g = g.remove_self_loop().add_self_loop()
    in_feats = g.ndata['features'].shape[1]

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    train_g.create_format_()
    val_g.create_format_()
    test_g.create_format_()
    # Pack data
    data = in_feats, n_classes, train_g, val_g, test_g

    max_inputs_length = args.batch_size

    b = args.batch_size
    fanout_list = list(map(int, args.fan_out.split(',')))
    fanout_list.reverse()
    for fanout in fanout_list:
        b *= fanout
        max_inputs_length += b

    args.num_epochs = 20
    args.worker_num = args.n_gpus
    args.number_of_nodes = g.number_of_nodes()
    args.cache_size_per_gpu = 3000000
    args.feature_dim = in_feats
    args.max_inputs_length = max_inputs_length
    args.samples_per_worker = 50
    args.layers = args.num_layers

    create_fifo(args.n_gpus)

    mp.set_start_method("spawn")

    procs = []

    for proc_id in range(args.n_gpus):
        p = mp.Process(target=receiving_process, args=(args, devices, data, proc_id))
        p.start()
        procs.append(p)

    p = mp.Process(target=caching_process, args=(args, devices, data))
    p.start()
    procs.append(p)

    for proc_id in range(args.n_gpus):
        p = mp.Process(target=training_process, args=(args, devices, data, proc_id))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


