import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import sklearn.linear_model as lm
import sklearn.metrics as skm

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
#from pyinstrument import Profiler

import socket
import pickle
from dgl.distributed import init_tensor_cache_engine, insert_partition_feature_tensor, remove_partition_feature_tensor, record_global2partition, record_global2local, gather_tensor_on_gpu
from dgl.distributed.kvstore import get_kvstore, GetPartDataRequest
from dgl.distributed import rpc 

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

            sampler = dgl.sampling.MultiLayerNeighborSampler([None])
            dataloader = dgl.sampling.NodeDataLoader(
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

def load_subtensor(seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    #batch_inputs = g.ndata['features'][input_nodes].to(device)
    #print(input_nodes)
    batch_inputs = gather_tensor_on_gpu(input_nodes)
    return batch_inputs

def recv_data(conn, length):
    res = bytes('',encoding='utf-8')
    #res = bytes('')
    while length != 0:
        data = conn.recv(length)
        length -= len(data)
        res += data
    return res



class DistSAGE(SAGE):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super(DistSAGE, self).__init__(in_feats, n_hidden, n_classes, n_layers,
                                       activation, dropout)

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

            sampler = PosNeighborSampler(g, [-1], dgl.distributed.sample_neighbors)
            print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
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


def cal_mrr(pos_score, neg_score, num_negs=1):
    aff = pos_score.reshape([len(pos_score), 1])
    neg_aff = neg_score.reshape([len(pos_score), num_negs])
    score = th.cat([aff, neg_aff], axis=-1)
    size = score.shape[-1]
    _, indices_of_ranks = th.topk(score, k=size)
    _, ranks = th.topk(-indices_of_ranks, k=size)
    ranks = (ranks + 1)[:, 0]
    mrr = th.mean(1.0 / ranks.float())
    return mrr


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph, num_negs):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        mrr = cal_mrr(pos_score, neg_score, num_negs)
        return loss, mrr


def generate_emb(model, g, inputs, batch_size, device):
    """
    Generate embeddings for each node
    g : The entire graph.
    inputs : The features of all the nodes.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)

    return pred

def compute_acc(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    
    We will fist train a LogisticRegression model using the trained embeddings,
    the training set, validation set and test set is provided as the arguments.

    The final result is predicted by the lr model.

    emb: The pretrained embeddings
    labels: The ground truth
    train_nids: The training set node ids
    val_nids: The validation set node ids
    test_nids: The test set node ids
    """

    emb = emb[np.arange(labels.shape[0])].cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    val_nids = val_nids.cpu().numpy()
    test_nids = test_nids.cpu().numpy()
    labels = labels.cpu().numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], labels[train_nids])

    pred = lr.predict(emb)
    eval_acc = skm.accuracy_score(labels[val_nids], pred[val_nids])
    test_acc = skm.accuracy_score(labels[test_nids], pred[test_nids])
    return eval_acc, test_acc

def run(args, device, data, sampler_socket_list):
    # Unpack data
    in_feats = data

    # Define model and optimizer
    model = DistSAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    #profiler = Profiler()
    #profiler.start()
    part_cached = [False for i in range(args.num_parts)]
    for epoch in range(args.num_epochs):
        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0

        step_time = []
        iter_t = []
        sample_t = []
        feat_copy_t = []
        forward_t = []
        backward_t = []
        update_t = []
        iter_tput = []

        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
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
                #length = conn.recv(8).decode()
                meta_length = int(conn.recv(1).decode())
                # receive sampling results via sockets
                length = int(conn.recv(meta_length).decode())
                res = recv_data(conn,int(length))
                (pos_graph, neg_graph, blocks) = pickle.loads(res)

                sample_t.append(tic_step - start)
                print("sample time {:.4f}".format(time.time() - start))

                copy_time = time.time()

                input_nodes = blocks[0].srcdata[dgl.NID]
                seeds = blocks[-1].dstdata[dgl.NID]
                batch_inputs = blocks[0].srcdata['features'].to(device)
                #batch_inputs = load_subtensor(seeds, input_nodes, device)

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.to(device) for block in blocks]
                # The nodes for input lies at the LHS side of the first block.
                # The nodes for output lies at the RHS side of the last block.
                # Load the input features as well as output labels
                #print("get feature takes {:.4f} s".format(time.time() - start))
                feat_copy_t.append(copy_time - tic_step)
                print("move graph and feature takes {:.4f} s".format(time.time() - copy_time))

                copy_time = time.time()
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss,mrr = loss_fcn(batch_pred, pos_graph, neg_graph, args.num_negs)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_t.append(forward_end - copy_time)
                backward_t.append(compute_end - forward_end)

                # Aggregate gradients in multiple nodes.
                optimizer.step()
                update_t.append(time.time() - compute_end)
                print("compute time {:.4f}".format(compute_end - copy_time))

                pos_edges = pos_graph.number_of_edges()
                neg_edges = neg_graph.number_of_edges()

                step_t = time.time() - start
                print("step total time: {:.4f} s".format(step_t))
                step_time.append(step_t)
                iter_tput.append(pos_edges / step_t)
                num_seeds += pos_edges
                if step % args.log_every == 0 and step != 0:
                    print('[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | time {:.3f} s' \
                            '| sample {:.3f} | copy {:.3f} | forward {:.3f} | backward {:.3f} | update {:.3f} | MRR {:.3f}'.format(
                        p_id, epoch, step, loss.item(), np.mean(iter_tput[3:]), np.sum(step_time[-args.log_every:]),
                        np.sum(sample_t[-args.log_every:]), np.sum(feat_copy_t[-args.log_every:]), np.sum(forward_t[-args.log_every:]),
                        np.sum(backward_t[-args.log_every:]), np.sum(update_t[-args.log_every:]),mrr))
                start = time.time()

            print('[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
                p_id, np.sum(step_time), np.sum(sample_t), np.sum(feat_copy_t), np.sum(forward_t), np.sum(backward_t), np.sum(update_t), num_seeds, num_inputs))

    # evaluate the embedding using LogisticRegression
    #if args.standalone:
    #    pred = generate_emb(model,g, g.ndata['features'], args.batch_size_eval, device)
    #else:
    #    pred = generate_emb(model.module, g, g.ndata['features'], args.batch_size_eval, device)
    #if g.rank() == 0:
    #    eval_acc, test_acc = compute_acc(pred, labels, global_train_nid, global_valid_nid, global_test_nid)
    #    print('eval acc {:.4f}; test acc {:.4f}'.format(eval_acc, test_acc))

    # sync for eval and test
    if not args.standalone:
        th.distributed.barrier()

    if not args.standalone:
        g._client.barrier()

        # save features into file
        if g.rank() == 0:
            th.save(pred, 'emb.pt')
    else:
        feat = g.ndata['features']
        th.save(pred, 'emb.pt')

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    args.kv_client = get_kvstore()
    args.kv_client.barrier()
    args.kv_client.barrier()
    args.kv_client.barrier()

    args.num_parts = 4
    args.steps_per_partition = 50000
    args.number_of_nodes = 1200000000 
    args.partition_num = 4 
    args.gpu_cache_size = 30000000
    args.feature_dim = 96 
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

    device = th.device('cuda:0')

    # Pack data
    in_feats = args.feature_dim
    data = in_feats
    run(args, device, data, sampler_socket_list)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_servers', type=int, default=1, help='Server count on each machine.')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--num_gpus', type=int, default=0, 
                        help="the number of GPU device. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=1,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--num_negs', type=int, default=2)
    parser.add_argument('--neg_share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    parser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")
    args = parser.parse_args()
    #assert args.num_workers == int(os.environ.get('DGL_NUM_SAMPLER')), \
    #'The num_workers should be the same value with num_samplers.'

    print(args)
    main(args)
