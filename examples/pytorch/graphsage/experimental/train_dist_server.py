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


def main(args):
    # for starting graph server, only invoke this function
    # this function will block until graph server receive "exit" message
    # graph server read config from bash ENV variables, so these function variaables are  dummy
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Graph Server')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--num_servers', type=int, default=4, help='The number of servers')
    parser.add_argument('--num_workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = parser.parse_args()

    print(args)
    main(args)
