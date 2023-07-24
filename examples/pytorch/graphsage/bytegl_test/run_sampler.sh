# export common ENV variables from comm_env.sh
source comm_env.sh

# sampler role for sampling process
export DGL_ROLE=sampler

# run sampler on each graph partition 
# and co-located with graph store 
python3.8 train_dist_sampler.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "ogb-papers100M_4.json" --num_client 1 --fan_out $FAN_OUT --num_sampling_processes $DGL_NUM_SAMPLING_PROCESSES --num_partitions $NUM_PARTITIONS --num_workers $NUM_WORKERS --samples_per_worker $SAMPLES_PER_WORKER
#python3.8 train_dist_gat_sampler.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "ogb-papers100M_4.json" --num_client 1 --fan_out $FAN_OUT --num_sampling_processes $DGL_NUM_SAMPLING_PROCESSES --num_partitions $NUM_PARTITIONS --num_workers $NUM_WORKERS --samples_per_worker $SAMPLES_PER_WORKER
#python3.8 train_dist_gcn_sampler.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "ogb-papers100M_4.json" --num_client 1 --fan_out $FAN_OUT --num_sampling_processes $DGL_NUM_SAMPLING_PROCESSES --num_partitions $NUM_PARTITIONS --num_workers $NUM_WORKERS --samples_per_worker $SAMPLES_PER_WORKER
#python3.8 train_dist_node_unsuper_sampler.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "ogb-papers100M_4.json" --num_client 1 --fan_out $FAN_OUT --num_sampling_processes $DGL_NUM_SAMPLING_PROCESSES --num_partitions $NUM_PARTITIONS --num_workers $NUM_WORKERS --samples_per_worker $SAMPLES_PER_WORKER
#python3.8 train_dist_node_unsuper_sampler.py --graph_name $GRAPH_NAME --ip_config $DGL_IP_CONFIG --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE  --part_config "ogb-papers100M_4.json" --num_client 1 --fan_out $FAN_OUT --num_workers 1
