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
import dgl.backend as backend

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import copy


import socket
import pickle

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

    def sample_blocks(self, seed_nodes):
        start = time.time()
        seed_nodes = th.LongTensor(np.asarray(seed_nodes))
        heads, tails = self.sample_neighbors(self.g, seed_nodes, 1, replace=True).edges()
        #heads, tails = self.g.find_edges(seed_nodes)
        n_nodes = len(heads)

        neg_tails = self.neg_sampler(self.num_negs * n_nodes).flatten()
        neg_heads = heads.view(-1, 1).expand(n_nodes, self.num_negs).flatten()

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

        input_nodes = blocks[0].srcdata[dgl.NID]
        blocks[0].srcdata['features'] = self.g.ndata['feat'][input_nodes]
            
        print("really sampling takes {:.4f} s".format(time.time() - start))

        # Pre-generate CSR format that it can be used in training directly
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


def run(args, device, data):
    # Unpack data
    train_nid, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], train_nid,
                              dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)
    #sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], th.LongTensor(np.arange(g.number_of_nodes())),
    #                          dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)


    print("begin dataloader")
    dataloader = dgl.distributed.DistDataLoader(
        dataset=copy.deepcopy(train_nid.numpy()),
        #dataset=train_eids.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)
    print("end dataloader")
    

    # use socket to send sampling results to worker
    address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[0].split(':')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("connect to worker")
    print(address, port)
    # sleep 5 seconds to wait worker 
    time.sleep(5)
    #sock.bind
    sock.connect((address, int(port)))
    print("send part_id ")
    sock.sendall(str(g.get_partition_book().partid).encode())


    print("begin send samples")
    # Training loop
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        start = time.time()
        step_time = []
        for step, tup in enumerate(dataloader):
            print("sample takes {:.4f} s".format(time.time() - start))

            # send sampling results to worker
            r = pickle.dumps(tup)
            length = str(len(r))
            print(length)
            meta_length = str(len(length))

            start = time.time()
            sock.sendall(meta_length.encode())
            sock.sendall(length.encode())
            sock.sendall(r)
            print("send taks {:.4f} s".format(time.time() - start))
            start = time.time()


def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    print("init dist graph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print('rank:', g.rank())

    print("get local nids")
    pb = g.get_partition_book()
    local_nid = pb._partid2nids[pb._part_id]
    #global_train_mask = dgl.data.load_tensors('train_mask.dgl')['train_mask']
    #local_train_mask = backend.gather_row(global_train_mask, local_nid)
    #local_train_nid = backend.boolean_mask(local_nid, local_train_mask)

    print('part {},  (local: {}))'.format(
        g.rank(),  len(local_nid)))

    device = th.device('cpu')
    # Pack data
    data = local_nid, g
    run(args, device, data)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Sampler')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--num_negs', type=int, default=2)
    parser.add_argument('--neg_share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    parser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")
    args = parser.parse_args()

    print(args)
    main(args)
