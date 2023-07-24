import os
os.environ['DGLBACKEND']='pytorch'
import multiprocessing
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
from dgl._ffi.ndarray import empty_shared_mem

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pyinstrument import Profiler

import queue
import threading
import socket
import pickle
from dgl.distributed.kvstore import get_kvstore, GetPartDataRequest
from dgl.distributed.shared_mem_utils import _to_shared_mem as tensor_to_shared_mem
from dgl.distributed import rpc
from dgl.heterograph import DGLBlock

def load_subtensor(labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    #batch_inputs = gather_tensor_on_gpu(input_nodes)
    batch_labels = th.randint(0,2,(len(seeds),)).long().to(device)
    #batch_labels = labels[seeds].long()
    return batch_inputs, batch_labels

def recv_data(conn, length):
    res = bytes('',encoding='utf-8')
    #res = bytes('')
    while length != 0:
        data = conn.recv(length)
        length -= len(data)
        res += data
    return res


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
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=activation, norm=norm,allow_zero_in_degree=True))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=activation, norm=norm,allow_zero_in_degree=True))
        # output layer
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, norm=norm,allow_zero_in_degree=True))
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
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def remove_shm_tensor(args, temp_queue):
    time.sleep(100)
    for step in range(args.num_samples_per_worker):
        data = temp_queue.get()
        time.sleep(1)
    

def receiving_process_recv_thread(args, sampler_socket_list, recv_queue, proc_id):
    temp_queue = queue.Queue()
    remove_thread = threading.Thread(target=remove_shm_tensor, args=(args, temp_queue))
    remove_thread.start()

    for step in range(args.num_samples_per_worker):
        conn = sampler_socket_list
        start = time.time()
        meta_length = int(conn.recv(1).decode())
        length = int(conn.recv(meta_length).decode())
        res = recv_data(conn,length)
        print("Proc_id {} recv takes {:.4f}s".format(proc_id, time.time() - start))
        blocks = pickle.loads(res)
        res = {}
        features = blocks[0].srcdata['features']
        res['feat_shape'] = features.shape
        features = tensor_to_shared_mem(features, "w{:d}p{:d}s{:d}_f".format(0, proc_id, step))
        res['feat'] = "w{:d}p{:d}s{:d}_f".format(0, proc_id, step)
        for i in range(args.num_layers):
            #blocks[i] = blocks[i].shared_memory("w{:d}s{:d}_b{:d}".format(0, step, i), formats=("csr","csc"))
            blocks[i] = blocks[i].shared_memory("w{:d}p{:d}s{:d}_b{:d}".format(0, proc_id, step, i), formats=("coo"))
            res['block_{}'.format(i)] = "w{:d}p{:d}s{:d}_b{:d}".format(0, proc_id, step, i)
        recv_queue.put(res)
        temp_queue.put((blocks, features))

def load_features_from_shm(res, device=None):
    shape = res['feat_shape']
    name = res['feat']
    data = empty_shared_mem(name, False, shape, "float32")
    dlpack = data.to_dlpack()
    features = dgl.backend.zerocopy_from_dlpack(dlpack)
    if device is not None:
        features = features.to(device)
    return features

def run(args, device, data, sampler_socket_list):
    in_feats, n_classes, labels = data


    # Define model and optimizer
    if args.model == "gat":
        model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, 4, F.relu, args.dropout)
    elif args.model == "gcn":
        model = GCN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    else:
        model = DistSAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if args.num_gpus > 1:
        model = th.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-4)
    #labels = labels.to(device)

    # Training loop
    iter_tput = []
    iter_time = []
    iter_sampling_time = []
    iter_feature_time = []
    iter_compute_time = []
    #profiler = Profiler()
    #profiler.start()
    part_cached = [False for i in range(args.num_parts)]
    print("num epochs: {:d}".format(args.num_epochs))
    args.num_epochs = 100
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0

        step_time = []
        step = 0
        epochs_seeds = 0
        # loop each partition
        for p_id in range(args.num_parts):
            #p_id = (p_id + 3) % args.num_parts
            # when feature tensor not in this partition, we send request and receive partition tensor
            #if part_cached[p_id] == False:
            #    #print("send get partition feature request")
            #    #rpc.send_request(p_id, GetPartDataRequest(args.feature_name))
            #    #print("start to receive")
            #    #recv_start = time.time()
            #    #res = rpc.recv_response()
            #    #print("recv partition feature takes {:.4f} s".format(time.time() - recv_start))
            #    data = dgl.data.load_tensors("node_feat96.dgl")['feat'].float()
            #    insert_partition_feature_tensor(p_id, data)
            #    part_cached[p_id] = True
            #    print("success insert part {:d}".format(p_id))

            #conn = sampler_socket_list[p_id]
            #conn = sampler_socket_list[0]
            for step  in range(args.steps_per_partition):
                step_start = time.time()
                tic_step = time.time()
                start = time.time()
                #length = conn.recv(7).decode()
                #meta_length = int(conn.recv(1).decode())
                # receive sampling results via sockets
                #length = int(conn.recv(meta_length).decode())
                #res = recv_data(conn,length)
                res = args.recv_queue[step % args.socket_num].get()
                blocks = []
                for i in range(args.num_layers):
                    block = dgl.hetero_from_shared_memory(res['block_{}'.format(i)])
                    block.is_block = True
                    blocks.append(block)
                    #blocks.append(DGLBlock(dgl.hetero_from_shared_memory(res['block_{}'.format(i)])))
                
                #blocks = pickle.loads(blocks)
                iter_sampling_time.append(time.time() - step_start)

                #print("recv takes {:.4f} s".format(time.time() - start))
                sample_time += tic_step - start
                #continue

                feature_tic = time.time()
                start = time.time()
                #input_nodes = blocks[0].srcdata[dgl.NID]
                #seeds = blocks[-1].dstdata[dgl.NID]
                seeds = th.Tensor(args.batch_size)
                #batch_inputs = blocks[0].srcdata['features'].to(device)
                #batch_labels = blocks[-1].dstdata['labels'].long().to(device)
                batch_inputs = load_features_from_shm(res, device)
                batch_labels = th.randint(0,2,(len(seeds),)).long().to(device)
                #batch_inputs, batch_labels = load_subtensor(labels, seeds, input_nodes, device)
                blocks = [block.to(device) for block in blocks]
                #print("move to gpu takes {:.4f} s".format(time.time() - start))


                #num_seeds += len(blocks[-1].dstdata[dgl.NID])
                #num_inputs += len(blocks[0].srcdata[dgl.NID])
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
                #print("compute takes {:.4f} s".format(compute_end - start))

                optimizer.step()
                update_time += time.time() - compute_end
                iter_time.append(time.time() - step_start)
                iter_feature_time.append(start - feature_tic)
                iter_compute_time.append(time.time() - start)    
                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(seeds) / (step_t))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    speed = len(seeds) * args.global_rank / np.mean(iter_time[-5*20:])
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | time {:.3f} s'.format(
                        p_id, epoch, step, loss.item(), acc.item(), speed, gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
                    print('Avg sampling time {:.4f}s | Feature time {:.4f}s | Compute time {:.4f}s.'.format(
                        np.mean(iter_sampling_time[-5*20:]), np.mean(iter_feature_time[-5*args.log_every:]), np.mean(iter_compute_time[-5*args.log_every:])))
                    #save_path = "bytegl_model_ckpt/model.iter-{:d}-{:d}-{:d}".format(epoch,p_id, step)
                    #th.save(model.state_dict(), save_path)
                start = time.time()
                step_start = time.time()

            toc = time.time()
            print('Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
                p_id, toc - tic, sample_time, copy_time, forward_time, backward_time, update_time, num_seeds, num_inputs))


            #if epoch % args.eval_every == 0 and epoch != 0:
            #    start = time.time()
            #    val_acc, test_acc = evaluate(model.module, g, g.ndata['features'],
            #                                 g.ndata['labels'], val_nid, test_nid, args.batch_size_eval, device)
            #    print('Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}'.format(g.rank(), val_acc, test_acc,
            #                                                                          time.time() - start))

    #profiler.stop()
    #print(profiler.output_text(unicode=True, color=True))

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    device = th.device('cuda:0')
    if args.num_gpus > 1:
        th.distributed.init_process_group(backend='gloo')
        print('rank:',th.distributed.get_rank(), ", world_size:", th.distributed.get_world_size())
        device = th.device('cuda:'+str(th.distributed.get_rank() % args.num_gpus))
    #g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    #print('rank:', g.rank())
    args.kv_client = get_kvstore()
    # a barrier in DistGraph
    args.kv_client.barrier()
    args.kv_client.barrier()
    args.kv_client.barrier()

    #num_parts = g.get_partition_book().num_partitions()
    args.num_parts = 4
    args.steps_per_partition = 100
    args.number_of_nodes = 1200000000
    args.partition_num = 4
    args.gpu_cache_size = 30000000
    args.feature_dim = 100
    args.feature_name = "node:feat"

    #address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[0].split(':')
    #address, port = "10.128.101.137:20003".split(':')
    address = "10.128.101.137"
    #port = str(20000 + th.distributed.get_rank())
    port = 20003
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print(address, port)
    sock.bind((address, int(port)))
    sock.listen(10)
    sampler_socket_list = [None for i in range(args.num_parts)]
    args.num_samples_per_worker = args.num_epochs * args.steps_per_partition
    print("accept socket")
    args.recv_queue = []
    socket_recv_threads = []
    args.socket_num = int(args.num_parts/th.distributed.get_world_size()) if args.num_gpus > 1 else args.num_parts

    for i in range(args.socket_num):
        conn, addr = sock.accept()
        # recv the partition id
        part_id = conn.recv(1).decode()
        print(part_id)
        # put connection in the right position
        sampler_socket_list[int(part_id)] = conn
        recv_queue = multiprocessing.Queue()
        args.recv_queue.append(recv_queue)
        socket_recv_thread = Process(target=receiving_process_recv_thread, args=(args, conn, recv_queue, i))
        socket_recv_thread.start()
        socket_recv_threads.append(socket_recv_thread)
    #init_tensor_cache_engine(args.number_of_nodes, args.partition_num, args.gpu_cache_size, args.feature_dim, args.feature_name, args.kv_client)


    #record_global2partition(dgl.data.load_tensors("node_map.dgl")["node_map"])

    #record_global2local(dgl.data.load_tensors("global2local.dgl")["global2local"])


    #labels = dgl.data.load_tensors("labels.dgl")['labels']
    labels = None
    
    #device = th.device('cuda:0')
    #labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    #n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    n_classes = 2

    print('#labels:', n_classes)

    # Pack data
    in_feats = args.feature_dim
    #in_feats = g.ndata['features'].shape[1]
    data = in_feats, n_classes, labels
    run(args, device, data, sampler_socket_list)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed supervised GraphSAGE')
    register_data_args(parser)
    parser.add_argument('--model', type=str, default="sage")
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--fan_out', type=str, default='5,10,15')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--global_rank', type=int, default=1)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=1,
                    help="Number of sampling processes. Use 0 for no extra process.")
    args = parser.parse_args()
    
    print(args)
    main(args)