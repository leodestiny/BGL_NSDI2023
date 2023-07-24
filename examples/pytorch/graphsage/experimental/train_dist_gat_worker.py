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
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def run(args, device, data, sampler_socket_list):
    # Unpack data
    in_feats, n_classes, labels = data

    # Define model and optimizer
    num_heads = 4
    args.num_layers=1
    model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #labels = labels.to(device)

    # Training loop
    iter_tput = []
    profiler = Profiler()
    profiler.start()
    epoch = 0
    part_cached = [False for i in range(args.num_parts)]
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        step_start = time.time()
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for p_id in range(args.num_parts):
            p_id = (p_id + 3) % args.num_parts
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
                
            conn = sampler_socket_list[p_id]

            for step in range(args.steps_per_partition):
                tic_step = time.time()

                start = time.time()
                # receive sampling results via sockets
                meta_length = int(conn.recv(1).decode())
                length = conn.recv(meta_length).decode()
                res = recv_data(conn,int(length))
                blocks = pickle.loads(res)
            
                #print("recv takes {:.4f} s".format(tic_step - start))
                sample_time += tic_step - start
                #print("sample time: {:4f} s".format(time.time() - start))
                sample_time += time.time() - start

                start = time.time()
                input_nodes = blocks[0].srcdata[dgl.NID]
                seeds = blocks[-1].dstdata[dgl.NID]
                batch_inputs = blocks[0].srcdata['features'].to(device)
                batch_labels = blocks[-1].dstdata['labels'].long().to(device)
                #batch_inputs, batch_labels = load_subtensor(labels, seeds, input_nodes, device)
                blocks = [block.to(device) for block in blocks]
                num_seeds += len(seeds)
                num_inputs += len(input_nodes)
                #print("move graph and feature {:.4f} s".format(time.time() - start))

                # Compute loss and prediction
                start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = F.nll_loss(batch_pred, batch_labels)	
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                optimizer.step()
                update_time += time.time() - compute_end

                #print("compute model : {:.4f} s".format(time.time() - start))

                step_t = time.time() - step_start
                step_time.append(step_t)
                iter_tput.append(len(seeds) / (step_t))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | time {:.3f} s'.format(
                        p_id, epoch, step, loss.item(), acc.item(), np.mean(iter_tput[-args.log_every:]), gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
                    #save_path = "model_ckpt/model.iter-{:d}-{:d}".format(epoch, step)
                    #th.save(model.state_dict(), save_path)
                step_start = time.time()

            toc = time.time()
            print('Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
                p_id, toc - tic, sample_time, copy_time, forward_time, backward_time, update_time, num_seeds, num_inputs))



    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    #print("init th distributed")
    #print("init dist graph")
    #g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    #print('rank:', g.rank())
    args.kv_client = get_kvstore()
    args.kv_client.barrier()
    args.kv_client.barrier()
    args.kv_client.barrier()

    args.num_parts = 4
    args.steps_per_partition = 50000
    args.number_of_nodes = 1200000000 
    args.partition_num = 4
    args.gpu_cache_size = 30000000
    args.feature_dim = 100
    args.feature_name = "node:feat"

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

    #labels = dgl.data.load_tensors("labels.dgl")['labels']
    labels = None 

    device = th.device('cuda:0')
    #labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    #n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    n_classes = 2
    print('#labels:', n_classes)


    # Pack data
    in_feats = args.feature_dim
    data = in_feats, n_classes, labels
    run(args, device, data, sampler_socket_list)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
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
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--fan_out', type=str, default='10')
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
