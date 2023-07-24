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
import socket
import pickle
from torch.nn.parallel import DistributedDataParallel

from dgl.heterograph import DGLBlock
from dgl import backend
argparser = argparse.ArgumentParser("multi-gpu training")
argparser.add_argument('--gpu', type=str, default='0',
    help="Comma separated list of GPU device IDs.")
argparser.add_argument('--dataset', type=str, default='ogb-product')
argparser.add_argument('--num-epochs', type=int, default=20)
argparser.add_argument('--num-hidden', type=int, default=256)
argparser.add_argument('--num-layers', type=int, default=3)
argparser.add_argument('--fan-out', type=str, default='5,10,15')
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--log-every', type=int, default=20)
argparser.add_argument('--eval-every', type=int, default=50)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=6,
    help="Number of sampling processes. Use 0 for no extra process.")
argparser.add_argument('--inductive', action='store_true',
    help="Inductive learning setting")
args = argparser.parse_args()

import byteps.torch as bps
bps.init()
torch.cuda.set_device(bps.local_rank())
cudnn.benchmark = True

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


# Define model and optimizer
model = SAGE(args.in_feats, args.num_hidden, args.n_classes, args.num_layers, F.relu, args.dropout)
model.cuda()
loss_fcn = nn.CrossEntropyLoss()
loss_fcn = loss_fcn.to(worker_id)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = bps.DistributedOptimizer(optimizer,
                                    named_parameters=model.named_parameters(),
                                    compression=compression)

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, args.in_feats)
    res = th.load("papers_2hop_sample")
    blocks = pickle.loads(res)
    input_nodes = blocks[0].srcdata[dgl.NID]
    seeds = blocks[-1].dstdata[dgl.NID]
    args.batch_size = len(seeds)
    batch_inputs = torch.rand(input_nodes, args.in_feats)
    data, batch_inputs= [block.int().cuda() for block in blocks], batch_inputs.cuda()
    datasets.append((data, batch_inputs))

data_index = 0    

def benchmark_step():
    global data_index

    (data, batch_inputs) = datasets[data_index%len(datasets)]
    input_nodes = blocks[0].srcdata[dgl.NID].cuda()
    seeds = blocks[-1].dstdata[dgl.NID].cuda()
    batch_labels = th.rand(seeds, args.n_classes)
    data_index += 1
    
    batch_pred = model(data, batch_inputs)
    loss = loss_fcn(batch_pred, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if bps.local_rank() != 0:
        return
    print(s, end='\n' if nl else '')
    sys.stdout.flush()


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, bps.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler & (bps.rank() == 0)

with torch.autograd.profiler.profile(enabled=enable_profiling, use_cuda=True) as prof:
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)


# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (bps.size(), device, bps.size() * img_sec_mean, bps.size() * img_sec_conf))


