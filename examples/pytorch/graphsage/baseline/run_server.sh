export DGL_DIST_MODE=distributed
export DGL_CONF_PATH="data/ogb-product.json"


export DGL_NUM_SERVER=1
export DGL_NUM_SAMPLER=1
# NUM_CLIENT = (1 + num_sampler) * num_workers
NUM_WORKER=4
export DGL_NUM_CLIENT=$[(1+$DGL_NUM_SAMPLER)*$NUM_WORKER]
echo "Number of client per server is $DGL_NUM_CLIENT"
export DGL_NUM_SAMPLER_PER_PART=1
export TRAINING_STEPS_PER_PARTITION=100
export DGL_SERVER_ID=$ARNOLD_ID
export DGL_IP_CONFIG=ip_config.txt
export DGL_ROLE=server
graph_name=ogb-product

pkill -9 python3.8
rm -r /dev/shm/*
python3.8 train_dist.py --graph_name ${graph_name} \
        --part_config ${DGL_CONF_PATH} \
        --num_gpus 1 \
        --ip_config ip_config.txt \
        --num_servers 1 \
        --num_epochs 10 \
        --batch_size 1000 \
        --num_workers 1 &

sleep 3