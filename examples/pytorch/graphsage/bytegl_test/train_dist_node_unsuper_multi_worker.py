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
import socket
import pickle
from torch.nn.parallel import DistributedDataParallel

from load_graph import load_reddit, load_ogb, inductive_split
from dgl.distributed.shared_mem_utils import _to_shared_mem as tensor_to_shared_mem
from dgl.distributed import start_caching_process, start_training_recv_thread, get_sample_result, start_receiving_children_thread, dispatch_shared_memory_task, wait_receiving_remove_shared_memory_thread
from dgl.distributed import dispatch_unsuper_shared_memory_task, set_training_dgl_kernel_stream
from dgl.heterograph import DGLBlock
from dgl import backend


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

def recv_data(conn, length):
    buf = bytearray(length)
    view = memoryview(buf)
    while length != 0:
        nbytes = conn.recv_into(view, length)
        length -= nbytes
        view = view[nbytes:]
    return buf

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

def caching_process(args, features):
    start_caching_process(args.worker_num, args.number_of_nodes, args.cache_size_per_gpu, args.feature_dim, args.max_inputs_length, features, args.num_samples_per_worker)

def training_process(args,  worker_id):

    if args.n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = args.n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=worker_id)
    th.cuda.set_device(worker_id)
    start_training_recv_thread(worker_id, args.max_inputs_length, args.feature_dim, args.layers, args.batch_size, th.zeros((0,)), args.num_samples_per_worker)

    # 0 to set no write buffer
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id),"wb", 0)

    # Define model and optimizer
    model = SAGE(args.in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
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
        for step in range(args.num_samples_per_worker):
            tic_step = time.time()

            sample_start = time.time()
            blocks, pos_graph, neg_graph, batch_inputs = get_sample_result(args.layers, use_label=False)
            sample_end = time.time()

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
            iter_time.append(batch_time)
            iter_sample.append(sample_end-sample_start)
            iter_compute.append(compute_end-compute_start)
            iter_tput.append((pos_edges + neg_edges) * args.worker_num / batch_time)
            if step % args.log_every == 0 and worker_id == 0:
                print('AVERAGE Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), (pos_edges + neg_edges) * args.worker_num / np.mean(iter_time[3:]),  np.mean(iter_time[3:]), np.mean(iter_sample[3:]), np.mean(iter_compute[3:])))
                print('CURRENT Epoch {:05d} | Step {:05d} | Loss {:.4f} | MRR {:.4f}| Speed (samples/sec) {:.4f} | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), mrr, iter_tput[-1], iter_time[-1], sample_end-sample_start, compute_end-compute_start))
            # notify receving process this sample is finished 
            train_fifo.write(worker_id.to_bytes(4, sys.byteorder))
            train_fifo.write(step.to_bytes(4,sys.byteorder))
	  
    if args.n_gpus > 1:
        th.distributed.barrier()
    return
            
def receiving_process_recv_thread(args, sampler_socket_list, recv_queue):
    for step in range(args.num_samples_per_worker):
        conn = sampler_socket_list[step % args.num_socket_per_worker]
        start = time.time()
        length = int.from_bytes(recv_data(conn, 4), sys.byteorder)
        res = recv_data(conn,length)
        print("recv {:.4f}MB takes {:.4f}s".format(length*1.0 / 2**20, time.time() - start))
        recv_queue.put(res)

def receiving_process(args, worker_id):
    num_moving_thread = 5
    start_receiving_children_thread(worker_id, args.layers, args.num_samples_per_worker, num_moving_thread)

    # connect to sampler socket
    address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[worker_id].split(':')
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind((address, int(port)))
    sock.listen(10)

    print("try to connect sampler")
    sampler_socket_list = []
    for i in range(args.num_socket_per_worker):
        conn, addr = sock.accept()
        print("accept connection ", i)
        sampler_socket_list.append(conn)

    # start socket recv thread
    recv_queue = queue.Queue()
    socket_recv_thread = threading.Thread(target=receiving_process_recv_thread, args=(args, sampler_socket_list, recv_queue))
    socket_recv_thread.start()

    # Receiving loop
    avg = 0
    step = -1
    for step in range(args.num_samples_per_worker):
        
        # receive sampling results via sockets
        start = time.time()
        res = recv_queue.get()
        pickle_start = time.time()
        pos_graph, neg_graph, blocks = pickle.loads(res)
        print("pickle takes {:.4f}s".format(time.time() - pickle_start))

        start = time.time()
        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]

        dispatch_unsuper_shared_memory_task(worker_id, step, blocks,  pos_graph, neg_graph,input_nodes, seeds)

    wait_receiving_remove_shared_memory_thread()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("worker scripts of distributed multi-gpu training")
    argparser.add_argument('--graph_name', type=str, help='graph name')
    argparser.add_argument('--id', type=int, help='the partition id')
    argparser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    argparser.add_argument('--part_config', type=str, help='The path to the partition config file')
    argparser.add_argument('--num_clients', type=int, help='The number of clients')
    argparser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    argparser.add_argument('--dataset', type=str, default='ogb-product')
    argparser.add_argument('--num_epochs', type=int, default=20)
    argparser.add_argument('--num_hidden', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3)
    argparser.add_argument('--fan_out', type=str, default='5,10,15')
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--log_every', type=int, default=1)
    argparser.add_argument('--eval_every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_workers', type=int, default=0,
        help="Number of workers in worker_machine")
    argparser.add_argument('--num_partitions', type=int, default=0,
        help="Number of partitions")
    argparser.add_argument('--num_samples_per_worker', type=int, default=0,
        help="Number of partitions")
    argparser.add_argument('--num_negs', type=int, default=1)
    argparser.add_argument('--neg_share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    argparser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")

    args = argparser.parse_args()

    args.n_gpus = args.num_workers

    args.feature_dim = 100

    # ogb-papers paprameter
    #args.in_feats = 128
    #args.n_classes = 172

    # ogb-products parameter
    args.in_feats = 100
    args.n_classes = 47

    #labels = dgl.data.load_tensors("labels.dgl")['labels']
    #labels = None

    #features = None
    features = dgl.data.load_tensors("features.dgl")['features']
    

    args.max_inputs_length = args.batch_size
    b = args.batch_size
    for fanout in list(map(int, args.fan_out.split(','))):
        b *= fanout
        args.max_inputs_length += b

    args.worker_num = args.n_gpus
    args.number_of_nodes = 2500000 
    args.cache_size_per_gpu = 3000000
    args.feature_dim = args.in_feats
    args.layers = args.num_layers

    if args.num_partitions > args.num_workers:
        args.num_socket_per_worker = int(args.num_partitions / args.num_workers)
    else:
        args.num_socket_per_worker = 1

    create_fifo(args.n_gpus)

    mp.set_start_method("spawn")

    procs = []

    for proc_id in range(args.n_gpus):
        p = mp.Process(target=receiving_process, args=(args, proc_id))
        p.start()
        procs.append(p)

    p = mp.Process(target=caching_process, args=(args, features))
    p.start()
    procs.append(p)

    for proc_id in range(args.n_gpus):
        p = mp.Process(target=training_process, args=(args, proc_id))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()




