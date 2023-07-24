# include some extra file in
export PYTHONPATH=$PYTHONPATH:..

# control the number of threads in dgl logic
# increasing the number may improve performance
export OMP_NUM_THREADS=1

# DGL support two distributed mode: standalone and distributed
# standalone mode is used for debug and test
export DGL_DIST_MODE=distributed

# the path of ip config file
# this file contains the ip address of each server
# config file can only contains ip address
# DGL use DEFAULT_PORT 30050
export DGL_IP_CONFIG="ip_config.txt"

# igraph partition config file
export DGL_CONF_PATH="data/ogb-product.json"
#export DGL_CONF_PATH="aweme_ad_4.json"

# total number of clients = 1 (worker) + s (sampler) * p (partition)
export DGL_NUM_CLIENT=1

# the number of server in one machine
# only one is master and others are backup
# they share underlying data by shared memory
export DGL_NUM_SERVER=1

# the number of samper in each partition
# this number influnces the performance of sampler
export DGL_NUM_SAMPLER_PER_PART=1


export TRAINING_STEPS_PER_PARTITION=100

# parameters for GNN training
GRAPH_NAME="ogb-product"
#GRAPH_NAME="aweme_ad_4"
BATCH_SIZE=1000
FAN_OUT=10,25
NUM_EPOCHS=100

