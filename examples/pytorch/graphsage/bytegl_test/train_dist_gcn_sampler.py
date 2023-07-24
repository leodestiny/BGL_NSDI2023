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


class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, device):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device

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

        #input_nodes = blocks[0].srcdata[dgl.NID]
        #seeds = blocks[-1].dstdata[dgl.NID]
        ##batch_inputs, batch_labels = load_subtensor(self.g, seeds, input_nodes, "cpu")
        #blocks[0].srcdata['features'] = self.g.ndata['feat'][input_nodes]
        #blocks[-1].dstdata['labels'] = th.randint(0,2,(len(seeds),)).long()

        return blocks


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
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors, device)

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid,
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=True)

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
