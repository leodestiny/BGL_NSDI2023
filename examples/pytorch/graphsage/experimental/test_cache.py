import dgl
import time
import os
from dgl.distributed import *
import torch as th

os.environ['DGL_DIST_MODE'] = 'distributed'
g = dgl.distributed.DistGraph('ip_config.txt','ogb-product','data/ogb-product.json')
pb = g.get_partition_book()
feat = g.ndata['features']
kv_client = feat.kvstore

init_tensor_cache_engine(g.number_of_nodes(), pb.get_node_size(), g.number_of_nodes(), feat.shape[1], kv_client)
record_cpu_tensor(kv_client._data_store['node:features'],pb.partid2nids(pb.partid))
record_global_partition(g.number_of_nodes(), pb)


#local_range = list(range(100000))
#print("get local feature")
#for i in range(10):
#    start = time.time()
#    t = g.ndata['features'][local_range]
#    print("local get takes {:.4f} s".format(time.time() - start))

s = 50000 
t = 100000
d = 50000


remote_range = th.arange(2000000,2050000)

print("get remote feature")
for i in range(48):
    start = time.time()
    res = gather_tensor_on_gpu(th.arange(s,t))
    s += d
    t += d
    #t = g.ndata['features'][remote_range]
    print("remote get takes {:.4f} s".format(time.time() - start))

