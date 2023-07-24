export DGL_DIST_MODE=distributed
graph_name=ogb-product
export DGL_CONF_PATH="data/${graph_name}.json"

export DGL_NUM_CLIENT=8
export DGL_NUM_SERVER=1
export DGL_NUM_SAMPLER=1
#export DGL_NUM_SAMPLER_PER_PART=1
export TRAINING_STEPS_PER_PARTITION=100
export DGL_SERVER_ID=$ARNOLD_ID
export DGL_ROLE=client
model=gat
# nnnodes: # of worker machines; nnproc_per_node: # of gpus per machine
python3.8 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=$ARNOLD_ID --master_addr=10.128.101.137 --master_port=20002 \
        train_dist_${model}.py --graph_name ${graph_name} \
        --part_config ${DGL_CONF_PATH} \
        --num_gpus 2 \
        --ip_config ip_config.txt \
        --num_servers 1 \
        --num_hidden 256 \
        --fan_out 5,10,15 \
        --num_layers 3 \
        --num_epochs 10 \
        --batch_size 1000 \
        --num_workers 1