import dgl
from load_graph import load_reddit, load_ogb, inductive_split
from dgl.distributed import *
import torch
import time
g, n_classes = load_ogb('ogbn-products')
feat = g.ndata['features']
init_tensor_cache_engine(g.number_of_nodes(), feat.shape[0], feat.shape[0], feat.shape[1])
record_cpu_tensor(feat, torch.arange(0,feat.shape[0]))
print("finish")
s = 0
t = 100000
d = 100000
r = torch.arange(0,200000)
r2 = torch.arange(5000,25000)
for i in range(1000):
    start = time.time()
    #a = gather_tensor_on_gpu(torch.arange(s,t))
    a = gather_tensor_on_gpu(r)
    s += d
    t += d
    print("takes {:.4f} s".format(time.time() - start))
#start = time.time()
#a = gather_tensor_on_gpu(r2)
#print("takes {:.4f} s".format(time.time() - start))
