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

import time
import math
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
import queue
import os
import sys
import threading
from torch.nn.parallel import DistributedDataParallel

from load_graph import load_reddit, load_ogb, inductive_split
from dgl.distributed.shared_mem_utils import _to_shared_mem as tensor_to_shared_mem
from dgl.distributed import start_caching_process, start_training_recv_thread, get_sample_result, start_receiving_children_thread, dispatch_shared_memory_task, wait_receiving_remove_shared_memory_thread
from dgl.heterograph import DGLBlock
from dgl.distributed import set_training_dgl_kernel_stream
from dgl import backend

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

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
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
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])


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
    start_training_recv_thread(worker_id, args.max_inputs_length, args.feature_dim, args.layers, args.batch_size, train_g.ndata['labels'], args.samples_per_worker)

    labels = train_g.ndata['labels']

    # 0 to set no write buffer
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id),"wb", 0)

    # Define model and optimizer
    num_heads = 4
    model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu, args.dropout)
    model = model.to(worker_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[worker_id], output_device=worker_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(worker_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iter_time = []
    iter_sample = []
    iter_compute = []
    iter_tput = []
    epoch = 0

    cuda_stream = th.cuda.Stream()
    set_training_dgl_kernel_stream(worker_id, cuda_stream.cuda_stream)
    #with th.cuda.profiler.profile():
    #    with th.autograd.profiler.emit_nvtx():
    with th.cuda.stream(cuda_stream):
        for step in range(args.samples_per_worker):
            #print("in step {:d}".format(step))
            tic_step = time.time()

            sample_start = time.time()
            blocks, batch_inputs, batch_labels = get_sample_result(args.layers, use_label=True)
            #blocks, batch_inputs, seeds = get_sample_result(args.layers, use_label=True)
            #batch_labels = labels[seeds].to(worker_id)
            #blocks, batch_inputs = get_sample_result(args.layers, use_label=True)
            sample_end = time.time()
            #print("convert time : {:.4f}s get ret_list: {:.4f}, block construct {:.4f}".format(time.time() - convert_start, convert_start - sample_start, block_end - block_start), batch_labels[0])
            #print(batch_inputs.shape)

            compute_start = time.time()
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            compute_end = time.time()


            batch_time = time.time() - tic_step
            #if batch_time < 0.1:
            iter_time.append(batch_time)
            iter_sample.append(sample_end-sample_start)
            iter_compute.append(compute_end-compute_start)
            iter_tput.append(batch_labels.shape[0] * args.worker_num / batch_time)
            if step % args.log_every == 0 and worker_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('AVERAGE Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), acc.item(), batch_labels.shape[0] * args.worker_num / np.mean(iter_time[3:]), gpu_mem_alloc, np.mean(iter_time[3:]), np.mean(iter_sample[3:]), np.mean(iter_compute[3:])))
                print('CURRENT Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), acc.item(), iter_tput[-1], gpu_mem_alloc, iter_time[-1], sample_end-sample_start, compute_end-compute_start))
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
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
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
        for input_nodes, seeds, blocks in dataloader:
            step += 1
            if step == args.samples_per_worker:
                break
            print("datalaoder takes {:.4f}s".format(time.time() - dataloader_start))
            
            #input_nodes = blocks[0].srcdata[dgl.NID]
            #seeds = blocks[-1].dstdata[dgl.NID]

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
            tmp_queue.put((worker_id, step, blocks, input_nodes, seeds))
            dataloader_start = time.time()


    for i in range(args.samples_per_worker):
            worker_id, step, blocks, input_nodes, seeds = tmp_queue.get()
            dispatch_shared_memory_task(worker_id, step, blocks, input_nodes, seeds)

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
    argparser.add_argument('--gpu', type=str,default='0',
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogb-product')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=1)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
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
    for fanout in list(map(int, args.fan_out.split(','))):
        b *= fanout
        max_inputs_length += b

    args.num_epochs = 20
    args.worker_num = args.n_gpus
    args.number_of_nodes = g.number_of_nodes()
    args.cache_size_per_gpu = 3000000
    args.feature_dim = in_feats
    args.max_inputs_length = max_inputs_length
    args.samples_per_worker = 100
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


