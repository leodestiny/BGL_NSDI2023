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

import socket
import pickle


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

        return blocks

def run(args, device, data):
    # Unpack data
    train_nid, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors, device)

    print("begin dataloader")
    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=False,
        drop_last=True)
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
        for step, blocks in enumerate(dataloader):
            print("sample takes {:.4f} s".format(time.time() - start))

            # send sampling results to worker
            r = pickle.dumps(blocks)
            print(len(r))

            start = time.time()
            sock.sendall(str(len(r)).encode())
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
    global_train_mask = dgl.data.load_tensors('train_mask.dgl')['train_mask']
    local_train_mask = backend.gather_row(global_train_mask, local_nid)
    local_train_nid = backend.boolean_mask(local_nid, local_train_mask)

    print('part {}, global_train_nids: {} (local: {}))'.format(
        g.rank(), len(th.nonzero(global_train_mask, as_tuple=True)[0]), len(local_train_nid)))

    device = th.device('cpu')
    # Pack data
    data = local_train_nid, g
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
    args = parser.parse_args()

    print(args)
    main(args)
