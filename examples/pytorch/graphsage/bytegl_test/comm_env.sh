# include some extra file in this directory
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
export DGL_CONF_PATH="ogb-product_2.json"
#export DGL_CONF_PATH="ogb-papers100M_4.json"
#export DGL_CONF_PATH="aweme_ad_4.json"

# total number of clients = 1 (worker) + s (sampler) * p (partition)
export DGL_NUM_CLIENT=10

# the number of server in one machine
# only first one is master and others are backup 
# they share underlying data by shared memory
export DGL_NUM_SERVER=1

# the number of sampling process in sampler 
# this number influnces the performance of sampler
export DGL_NUM_SAMPLING_PROCESSES=4


# graph name in shared memory used by DGL
export GRAPH_NAME="ogb-product_2"
#export GRAPH_NAME="aweme_ad_4"
#export GRAPH_NAME="ogb-papers100M_4"

# parameters for distributed training
# the number of partitions of entire graph 
export NUM_PARTITIONS=2
# the number of workers in worker machine
export NUM_WORKERS=1

# parameters for GNN training
export BATCH_SIZE=1000
export FAN_OUT=5,10,15
export NUM_EPOCHS=100
export SAMPLES_PER_WORKER=200


export ARNOLD_WORKER_HOSTS=10.130.23.77:9000,10.130.23.77:9001,10.130.23.77:9002,10.130.23.77:9003,10.130.23.77:9004,10.130.23.77:9005,10.130.23.77:9006,10.130.23.77:9007,10.130.23.77:9008

