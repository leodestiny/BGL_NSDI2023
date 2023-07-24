import os
os.environ['DGLBACKEND']='pytorch'
import argparse, time, math
import numpy as np

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
import threading

import socket
import pickle
import copy
import sys
import queue

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

        #input_nodes = blocks[0].srcdata[dgl.NID]
        #blocks[0].srcdata['features'] = self.g.ndata['feat'][input_nodes]
            
        #print("really sampling takes {:.4f} s".format(time.time() - start))

        # Pre-generate CSR format that it can be used in training directly
        return pos_graph, neg_graph, blocks



def sending_thread(args, sock, sending_queue, sending_thread_finish_event):
    for i in range(args.num_samples_per_sampler):
        r = sending_queue.get()
        start = time.time()
        sock.sendall(len(r).to_bytes(4, sys.byteorder))
        sock.sendall(r)
        print("send takes {:.4f} s".format(time.time() - start))
    sending_thread_finish_event.set()



def pickling_thread(args, prefetching_queue, sending_queue):
    print("pickling is start")
    for i in range(args.num_samples_per_sampler):
        blocks = prefetching_queue.get()
        r = pickle.dumps(blocks)
        sending_queue.put(r)

def prefetching_thread(args, dataloader, prefetching_queue):
    sample_cnt = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        for step, blocks in enumerate(dataloader):
            sample_cnt += 1
            if sample_cnt > args.num_samples_per_sampler:
                return
            prefetching_queue.put(blocks)
            print("get from dataloader takes {:.4f} s".format(time.time() - start))
            start = time.time()



def sampler_main(args, sampler_local_id):

    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_sampling_processes)
    print("init dist graph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print('rank:', g.rank())

    pb = g.get_partition_book()
    train_nid = pb._partid2nids[pb._part_id].clone().detach()

    device = th.device('cpu')

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], train_nid,
                              dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)

    print("start dataloader")
    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid,
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=True)

    print("start dataloader done")

    # calculate the target worker id of this sampler
    sampler_global_id = int(os.environ['ARNOLD_ID']) * args.num_samplers_per_part + sampler_local_id
    if args.num_workers >= args.num_partitions:
        target_worker_id = sampler_global_id
    else:
        target_worker_id = sampler_global_id % args.num_workers
    address, port = os.environ['ARNOLD_WORKER_HOSTS'].split(',')[target_worker_id].split(':')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("connect to worker", address, port)
    # sleep 5 seconds to wait worker 
    # time.sleep(5)
    sock.connect((address, int(port)))

    prefetching_queue = queue.Queue()
    sending_queue = queue.Queue()

    sampler_prefetching_thread = threading.Thread(target=prefetching_thread, args=(args, dataloader, sending_queue))
    sampler_prefetching_thread.start()

    #print("start pickling")
    #sampler_pickling_thread = threading.Thread(target=pickling_thread, args=(args, prefetching_queue, sending_queue))
    #sampler_pickling_thread.start()

    #print("start sending")
    #sampler_sending_thread = threading.Thread(target=sending_thread, args=(args, sock, sending_queue, sending_thread_finish_event))
    #sampler_sending_thread.start()

    for i in range(args.num_samples_per_sampler):
        r = sending_queue.get()
        start = time.time()
        sock.sendall(len(r).to_bytes(4, sys.byteorder))
        sock.sendall(r)
        print("send takes {:.4f} s".format(time.time() - start))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Sampler')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--num_servers', type=int, default=1, help='The number of servers')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--fan_out', type=str, default='5,10,15')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_sampling_processes', type=int, default=4,
            help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--num_partitions', type=int, default=4,
            help="Number of partition")
    parser.add_argument('--num_workers', type=int, default=1,
            help="Number of workers in worker_machine")
    parser.add_argument('--samples_per_worker', type=int, default=1,
            help="Number of samples each worker should process")
    parser.add_argument('--num_negs', type=int, default=1)
    parser.add_argument('--neg_share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    parser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")

    args = parser.parse_args()

    if args.num_workers > args.num_partitions:
        args.num_samplers_per_part =  int(args.num_workers / args.num_partitions)
        args.num_samples_per_sampler = args.samples_per_worker
    else:
        args.num_samplers_per_part = 1
        args.num_samples_per_sampler = int(args.samples_per_worker * args.num_workers / args.num_partitions)
    print(args)

    mp.set_start_method("spawn")

    procs = []
    for i in range(args.num_samplers_per_part):
        p = mp.Process(target=sampler_main, args=(args,i))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
