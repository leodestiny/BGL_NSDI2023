# export common ENV variables from comm_env.sh
source comm_env.sh

# default role for training process
export DGL_ROLE=default

# run worker in 
python3.8 train_dist_worker.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "data/ogb-product.json"  --fan_out $FAN_OUT --gpu 0 
#python3.8 train_dist_gat_worker.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "data/ogb-product.json"  --fan_out $FAN_OUT --num_gpus 0 
#python3.8 train_dist_node_unsuper_worker.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "data/ogb-product.json"  --fan_out $FAN_OUT --num_gpus 0 
#python3.8 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=5000 train_dist.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --batch_size $BATCH_SIZE --fan_out $FAN_OUT --num_gpus 0 --num_epochs $NUM_EPOCHS
#python3.8 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=5000 train_dist_node_unsupervised.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --batch_size $BATCH_SIZE --fan_out $FAN_OUT --num_gpus 0 --num_epochs $NUM_EPOCHS
