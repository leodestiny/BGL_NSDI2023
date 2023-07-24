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
from dgl.distributed import start_caching_process, start_training_recv_thread, get_sample_result
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
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

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
    res = bytes('',encoding='utf-8')
    #res = bytes('')
    while length != 0:
        data = conn.recv(length)
        length -= len(data)
        res += data
    return res

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

def caching_process(args, device, features):
    start_caching_process(args.worker_num, args.number_of_nodes, args.cache_size_per_gpu, args.feature_dim, args.max_inputs_length, features, args.samples_per_worker)

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
    start_training_recv_thread(worker_id, args.max_inputs_length, args.feature_dim, args.layers, args.batch_size, labels, args.samples_per_worker)

    # 0 to set no write buffer
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id),"wb", 0)

    # Define model and optimizer
    model = SAGE(args.in_feats, args.num_hidden, args.n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(worker_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[worker_id], output_device=worker_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(worker_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iter_time = []
    iter_tput = []
    epoch = 0

    cuda_stream = th.cuda.Stream()
    #with th.cuda.profiler.profile():
    #    with th.autograd.profiler.emit_nvtx():
    with th.cuda.stream(cuda_stream):
        for step in range(args.samples_per_worker):
            #print("in step {:d}".format(step))
            tic_step = time.time()

            sample_start = time.time()
            blocks, batch_inputs, batch_labels = get_sample_result(args.layers, use_label=True)
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


            batch_time = time.time() - tic_step;
            iter_time.append(batch_time)
            iter_tput.append(batch_labels.shape[0] * args.worker_num / batch_time)
            if step % args.log_every == 0 and worker_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('AVERAGE Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | Batch time {:.4f}s'.format(
                    epoch, step, loss.item(), acc.item(), batch_labels.shape[0] * args.worker_num / np.mean(iter_time[3:]), gpu_mem_alloc, np.mean(iter_time[3:])))
                print('CURRENT Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | Batch time {:.4f}s | Sample time {:.4f}s | Compute time {:.4f}s'.format(
                    epoch, step, loss.item(), acc.item(), iter_tput[-1], gpu_mem_alloc, iter_time[-1], sample_end-sample_start, compute_end-compute_start))
            # notify receving process this sample is finished 
            train_fifo.write(worker_id.to_bytes(4, sys.byteorder))
            train_fifo.write(step.to_bytes(4,sys.byteorder))
	  
    if args.n_gpus > 1:
        th.distributed.barrier()
    return

def receiving_process_remove_shared_mem_thread(args, train_fifo, sampling_queue):

    for i in range(args.samples_per_worker):
        t_wd = int.from_bytes(read_nbytes(train_fifo, 4), sys.byteorder)
        t_sx = int.from_bytes(read_nbytes(train_fifo, 4), sys.byteorder)

        worker_id,step, blocks, input_nodes, seeds = sampling_queue.get()
        if worker_id != t_wd or t_sx != step:
            print("out of order")

        del blocks
        del input_nodes
        del seeds



#### Entry point
def receiving_process(args, worker_id):
    ## Unpack data
    #in_feats, n_classes, train_g, val_g, test_g = data
    #train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    #val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    #test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    ## Create PyTorch DataLoader for constructing blocks
    #sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #    [int(fanout) for fanout in args.fan_out.split(',')])
    #dataloader = dgl.dataloading.NodeDataLoader(
    #    train_g,
    #    train_nid,
    #    sampler,
    #    batch_size=args.batch_size,
    #    shuffle=True,
    #    drop_last=True,
    #    num_workers=args.num_workers)
    

    # connect 
    address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[worker_id].split(':')
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind((address, int(port)))
    sock.listen(10)

    sampler_socket_list = []
    for i in range(args.num_socket_per_worker):
        conn, addr = sockt.accpet()
        sampler_socket_list.append(conn)
        

    max_pending_samples = 10
    sampling_queue = queue.Queue(max_pending_samples)

    # 0 to set no write buffer
    recv_fifo = open("/tmp/recv_fifo_w{:d}".format(worker_id),"wb", 0)


    # training process uses train_fifo to notify receiving process can release the shared memory
    # receiving process reads info from train_fifo
    train_fifo = open("/tmp/train_fifo_w{:d}".format(worker_id), "rb", 0)


    remove_shared_mem_thread = threading.Thread(target=receiving_process_remove_shared_mem_thread, args=(args, train_fifo, sampling_queue))
    remove_shared_mem_thread.start()



    # Training loop
    avg = 0
    step = -1
    for step in range(args.samples_per_worker):
        conn = sampler_socket_list[step % args.num_socket_per_worker]

        meta_length = int(conn.recv(1).decode())
        # receive sampling results via sockets
        length = int(conn.recv(meta_length).decode())
        res = recv_data(conn,length)
        blocks = pickle.loads(res)

        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]
        input_nodes = tensor_to_shared_mem(input_nodes, "w{:d}s{:d}_inputs".format(worker_id, step))
        seeds = tensor_to_shared_mem(seeds, "w{:d}s{:d}_seeds".format(worker_id, step))

        # copy blocks into shared memory
        for i in range(args.num_layers):
            # we create all formats for each block 
            # the graph name of each block is w0s0_b0, which means 'worker 0, step 0, block 0'
            blocks[i] = blocks[i].shared_memory("w{:d}s{:d}_b{:d}".format(worker_id, step, i), formats=("csc"))


        sampling_queue.put((worker_id,step, blocks, input_nodes, seeds))

        # write info to recv_fifo
        # four parameter: worker_id, step, length of input_nodes, length of seeds
        # each parameter use 4 bytes (as int) with system byteorder, little endian in linux
        recv_fifo.write(worker_id.to_bytes(4,sys.byteorder))
        recv_fifo.write(step.to_bytes(4,sys.byteorder))
        recv_fifo.write(input_nodes.shape[0].to_bytes(4,sys.byteorder))
        recv_fifo.write(seeds.shape[0].to_bytes(4,sys.byteorder))
#            print("successfully put step {:d} in sampling queue size {:d}".format(step,input_nodes.shape[0]))

            
    remove_shared_mem_thread.join()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser("worker scripts of distributed multi-gpu training")
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
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of workers in worker_machine")
    argparser.add_argument('--num-partitions', type=int, default=0,
        help="Number of partitions")
    argparser.add_argument('--num-samples-per-worker', type=int, default=0,
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

    features = None
    

    args.max_inputs_length = args.batch_size
    b = args.batch_size
    for fanout in list(map(int, args.fan_out.split(','))):
        b *= fanout
        args.max_inputs_length += b

    args.worker_num = args.n_gpus
    args.number_of_nodes = 100000000 
    args.cache_size_per_gpu = 3000000
    args.feature_dim = args.in_feats
    args.layers = args.num_layers

    if args.num_partitions > args.num_workers:
        args.num_socket_per_worker = args.num_partitions / args.num_workers
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


