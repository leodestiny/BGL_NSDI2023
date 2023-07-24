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
from dgl.nn.pytorch import GraphConv
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
import socket
import pickle
from torch.nn.parallel import DistributedDataParallel

from load_graph import load_reddit, load_ogb, inductive_split
from dgl.distributed.shared_mem_utils import _to_shared_mem as tensor_to_shared_mem
from dgl.distributed import start_caching_process, start_training_recv_thread, get_sample_result, start_receiving_children_thread, dispatch_shared_memory_task, wait_receiving_remove_shared_memory_thread
from dgl.distributed import set_training_dgl_kernel_stream
from dgl.heterograph import DGLBlock
from dgl import backend

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        norm = 'right'
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=norm))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=norm))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm=norm))
        #self.dropout = nn.Dropout(p=dropout)

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
                h = layer(block, (h, h_dst))
                #h = self.dropout(h)
            else:
                h = layer(block, (h, h_dst))
        return h.log_softmax(dim=-1)


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

def training_process(args, labels, worker_id):

    if args.n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = args.n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=worker_id)
    th.cuda.set_device(worker_id)
    start_training_recv_thread(worker_id, args.max_inputs_length, args.feature_dim, args.layers, args.batch_size, labels, args.num_samples_per_worker)

    # 0 to set no write buffer
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id),"wb", 0)

    # Define model and optimizer
    model = GCN(args.in_feats, args.num_hidden, args.n_classes, args.num_layers, F.relu, args.dropout)
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
    # with th.cuda.profiler.profile():
    #    with th.autograd.profiler.emit_nvtx():
    with th.cuda.stream(cuda_stream):
    #if True:
        for step in range(args.num_samples_per_worker):
            #print("in step {:d}".format(step))
            tic_step = time.time()

            sample_start = time.time()
            blocks, batch_inputs, batch_labels = get_sample_result(args.layers, use_label=True)
            sample_end = time.time()

            compute_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            compute_end = time.time()


            batch_time = time.time() - tic_step
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

def receiving_process_recv_thread(args, sampler_socket_list, recv_queue):
    for step in range(args.num_samples_per_worker):
        conn = sampler_socket_list[step % args.num_socket_per_worker]
        start = time.time()
        length = int.from_bytes(recv_data(conn, 4), sys.byteorder)
        res = recv_data(conn,length)
        #print("recv takes {:.4f}s".format(time.time() - start))
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
        
        step_start = time.time()
        # receive sampling results via sockets
        start = time.time()
        res = recv_queue.get()
        #print("get from recv_queue takes {:.4f}s".format(time.time() -start))
        pickle_start = time.time()
        blocks = pickle.loads(res)
        #print("pickle takes {:.4f}s".format(time.time() - pickle_start))
        #print("entire takes {:.4f}s".format(time.time() - start))

        start = time.time()
        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]

        dispatch_shared_memory_task(worker_id, step, blocks, input_nodes, seeds)
        #print("dispatching shared memory task takes {:.4f}s".format(time.time() - start))

    wait_receiving_remove_shared_memory_thread()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
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
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_workers', type=int, default=0,
        help="Number of workers in worker_machine")
    argparser.add_argument('--num_partitions', type=int, default=0,
        help="Number of partitions")
    argparser.add_argument('--num_samples_per_worker', type=int, default=0,
        help="Number of partitions")
    args = argparser.parse_args()

    args.n_gpus = args.num_workers

    args.feature_dim = 100

    # ogb-papers paprameter
    #args.in_feats = 128
    #args.n_classes = 172

    # ogb-products parameter
    args.in_feats = 100
    args.n_classes = 47

    labels = dgl.data.load_tensors("labels.dgl")['labels']
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
        p = mp.Process(target=training_process, args=(args, labels, proc_id))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()



