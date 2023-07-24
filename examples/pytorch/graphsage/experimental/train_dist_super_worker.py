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


import socket
import pickle
from dgl.distributed import init_tensor_cache_engine, insert_partition_feature_tensor, remove_partition_feature_tensor, record_global2partition, record_global2local, gather_tensor_on_gpu
from dgl.distributed.kvstore import get_kvstore, GetPartDataRequest
from dgl.distributed import rpc 

def load_subtensor(labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    #batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_inputs = gather_tensor_on_gpu(input_nodes)
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

def recv_data(conn, length):
    res = bytes('',encoding='utf-8')
    #res = bytes('')
    while length != 0:
        data = conn.recv(length)
        length -= len(data)
        res += data
    return res


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
        nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
                                           g.get_partition_book(), force_even=True)
        y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_hidden), th.float32, 'h',
                                       persistent=True)
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_classes),
                                               th.float32, 'h_last', persistent=True)

            sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors, device)
            print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DistDataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False)

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

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

def run(args, device, data, sampler_socket_list):
    in_feats, n_classes, labels = data


    # Define model and optimizer
    model = DistSAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    labels = labels.to(device)

    # Training loop
    iter_tput = []
    profiler = Profiler()
    profiler.start()
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
            # when feature tensor not in this partition, we send request and receive partition tensor
            if part_cached[p_id] == False:
                #print("send get partition feature request")
                rpc.send_request(p_id, GetPartDataRequest(args.feature_name))
                #print("start to receive")
                recv_start = time.time()
                res = rpc.recv_response()
                #print("recv partition feature takes {:.4f} s".format(time.time() - recv_start))
                insert_partition_feature_tensor(p_id, res.data)
                part_cached[p_id] = True
                
            conn = sampler_socket_list[p_id]

            for step  in range(args.steps_per_partition):

                tic_step = time.time()
                start = time.time()
                # receive sampling results via sockets
                length = conn.recv(7).decode()
                res = recv_data(conn,int(length))
                blocks = pickle.loads(res)
            
                #print("recv takes {:.4f} s".format(tic_step - start))
                sample_time += tic_step - start


                start = time.time()
                input_nodes = blocks[0].srcdata[dgl.NID]
                seeds = blocks[-1].dstdata[dgl.NID]
                batch_inputs, batch_labels = load_subtensor(labels, seeds, input_nodes, device)
                blocks = [block.to(device) for block in blocks]
                #print("move to gpu takes {:.4f} s".format(time.time() - start))


                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
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

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(seeds) / (step_t))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | time {:.3f} s'.format(
                        p_id, epoch, step, loss.item(), acc.item(), np.mean(iter_tput[-args.log_every:]), gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
                    save_path = "bytegl_model_ckpt/model.iter-{:d}-{:d}-{:d}".format(epoch,p_id, step)
                    th.save(model.state_dict(), save_path)
                start = time.time()

            toc = time.time()
            print('Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
                p_id, toc - tic, sample_time, copy_time, forward_time, backward_time, update_time, num_seeds, num_inputs))


            #if epoch % args.eval_every == 0 and epoch != 0:
            #    start = time.time()
            #    val_acc, test_acc = evaluate(model.module, g, g.ndata['features'],
            #                                 g.ndata['labels'], val_nid, test_nid, args.batch_size_eval, device)
            #    print('Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}'.format(g.rank(), val_acc, test_acc,
            #                                                                          time.time() - start))

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    #g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    #print('rank:', g.rank())
    args.kv_client = get_kvstore()
    # a barrier in DistGraph
    args.kv_client.barrier()
    args.kv_client.barrier()
    args.kv_client.barrier()

    #num_parts = g.get_partition_book().num_partitions()
    args.num_parts = 4
    args.steps_per_partition = 1000
    args.number_of_nodes = 111059956
    args.partition_num = 4
    args.gpu_cache_size = 30000000
    args.feature_dim = 128
    args.feature_name = "node:features"

    address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[0].split(':')
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print(address, port)
    sock.bind((address, int(port)))
    sock.listen(10)
    sampler_socket_list = [None for i in range(args.num_parts)] 

    print("accept socket")
    for i in range(args.num_parts):
        conn, addr = sock.accept()
        # recv the partition id
        part_id = conn.recv(1).decode()
        print(part_id)
        # put connection in the right position
        sampler_socket_list[int(part_id)] = conn

    init_tensor_cache_engine(args.number_of_nodes, args.partition_num, args.gpu_cache_size, args.feature_dim, args.feature_name, args.kv_client)

    
    record_global2partition(dgl.data.load_tensors("node_map.dgl")["node_map"])

    record_global2local(dgl.data.load_tensors("global2local.dgl")["global2local"])

    labels = dgl.data.load_tensors("labels.dgl")['labels'].long()

    device = th.device('cuda:0')
    #labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
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
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=1,
                    help="Number of sampling processes. Use 0 for no extra process.")
    args = parser.parse_args()

    print(args)
    main(args)
