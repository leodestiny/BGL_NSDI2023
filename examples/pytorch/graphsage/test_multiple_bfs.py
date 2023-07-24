import dgl
import torch as th
import load_graph
import time
g = load_graph.load_ogb("ogbn-products")[0]
start = time.time()
g.create_format_()
print("creating format takes {:.4f}s".format(time.time() - start))
mask = g.ndata['train_mask']
mask_nonzero_count = len(th.nonzero(mask))
pass_num = 1
start = time.time()
bfs_seq = dgl.traversal.multiple_bfs_node_generator_with_mask(g, th.randint(g.number_of_nodes(),(pass_num,)),mask, mask_nonzero_count,pass_num)
print("generate {:d} bfs_seq takes {:.4f} s".format(pass_num, time.time() - start))
print(bfs_seq.shape)
