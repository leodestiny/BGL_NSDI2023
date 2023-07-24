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
from dgl.nn.pytorch import GraphConv
import time
import math
import argparse
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel
import tqdm
import traceback
from utils import thread_wrapped_func
from load_graph import load_reddit, load_ogb, inductive_split

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
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
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
            else:
                h = layer(block, (h, h_dst))
        return h.log_softmax(dim=-1)


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, labels, seeds, input_nodes, dev_id):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels

#### Entry point

def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # Split train_nid
    train_nid = th.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = GCN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    iter_time = []
    iter_sampling_time = []
    iter_feature_time = []
    iter_compute_time = []
    avg_speed = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        sampling_tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            if proc_id == 0:
                tic_step = time.time()
                sampling_toc = time.time()
            # Load the input features as well as output labels
            feat_tic = time.time()
            #batch_inputs, batch_labels = load_subtensor(train_g, train_g.ndata['labels'], seeds, input_nodes, dev_id)
            blocks = [block.int().to(dev_id) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']
            feat_toc = time.time()

            # Compute loss and prediction
            compute_tic = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            #loss = F.nll_loss(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            compute_toc = time.time()
            if proc_id == 0:
                #iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
                iter_time.append(time.time() - sampling_tic)
                iter_tput.append(len(seeds) * n_gpus / (time.time() - sampling_tic))
                iter_sampling_time.append(sampling_toc - sampling_tic)
                iter_feature_time.append(feat_toc - feat_tic)
                iter_compute_time.append(compute_toc - compute_tic)
                speed = len(seeds) * n_gpus / np.mean(iter_time[3:])
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch, step, loss.item(), acc.item(), speed, th.cuda.max_memory_allocated() / 1000000))
                if step > 3:
                    feat_speed = (len(input_nodes) + len(seeds)) * 100 * 4 / np.mean(iter_feature_time[3:]) / 1024 / 1024 / 1024
                    print('Avg sampling time {:.4f}s | Feature time {:.4f}s | Feature speed {:.4f} GB/s | Compute time {:.4f}s.'.format(
                        np.mean(iter_sampling_time[3:]), np.mean(iter_feature_time[3:]), feat_speed, np.mean(iter_compute_time[3:])))
            sampling_tic = time.time()
            
        if n_gpus > 1:
            #th.distributed.barrier()
            pass
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
        help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='ogb-product')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=128)
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

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogb-product':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')
    print('Total edges before adding self-loop {}'.format(g.number_of_edges()))
    g = g.remove_self_loop().add_self_loop()
    print('Total edges after adding self-loop {}'.format(g.number_of_edges()))
    #g, n_classes = load_reddit()
    # Construct graph
    g = dgl.as_heterograph(g)
    in_feats = g.ndata['features'].shape[1]

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    #train_g.create_format_()
    #val_g.create_format_()
    #test_g.create_format_()
    # Pack data
    data = in_feats, n_classes, train_g, val_g, test_g
    #args.num_layers = 3
    #args.fan_out = '5,10,15'
    #args.batch_size = 333
    mp.set_start_method("spawn")
    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run,
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
