//Copyright 2018 Husky Data Lab, CUHK
//Authors: Hongzhi Chen, Miao Liu

using atomic_ptr_t = std::shared_ptr<std::atomic<int>>;
// @yrchen: Read label file
/*
vector<int> load_labels(const char* inpath)
{
	hdfsFS fs = get_hdfs_fs();
	hdfsFile in = get_r_handle(inpath, fs);
	LineReader reader(fs, in);
	vector<int> labels;
	while (true)
	{
		reader.read_line();
		if (!reader.eof()){
			int label = atoi(reader.get_line());
			// cout << "read label line " << label << endl;
			labels.push_back(label);
		}
		else
			break;
	}
	hdfsCloseFile(fs, in);
	hdfsDisconnect(fs);
	int label_count = 0;
	for(size_t i = 0; i < labels.size(); i++) {
		if (labels[i] == 1)
			label_count ++;
	}
	cout << "Load labels " << label_count << " out of " << labels.size() << " nodes." << endl;
	return labels;
}
// @yrchen: Check if the given vertex is labeled node
// for workload balancing
int IsLabelVertex(const vector<int> &labels, VertexID vid, bool synthetic=false) {
    if(synthetic) {
        if (vid % 4 == 0) {
            return 1;
        } else {
            return 0;
        }
    }
	return labels[vid];
}
*/
template<class BDGPartVertexT>
void BDGPartitioner<BDGPartVertexT>::block_assign()
{
	// @yrchen: read labels of vertices
	string inpath = "/user/yrchen/euler2gminer/gminer_input_data/label_dir/product_labels.txt";
	if (const char* env_p = getenv("LABEL_PATH")) {
		inpath = string(env_p);
	}
	cout << "Loading labels from path " << inpath << endl;
	//vector<int> labels = load_labels(inpath.c_str());
    vector<int> labels;
    bool synthetic = true;
	//collect Voronoi cell size and neighbors' bids
	BInfoMap b_info_map;   //key: color   value: blockInfo (color, size, neighbor bid)
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		VertexID color = (*it)->value().color;
		vector<kvpair>& nbs = (*it)->value().nbs_info;
		BIter b_iter = b_info_map.find(color);
		if (b_iter == b_info_map.end())
		{
			blockInfo b_info(color, 1);
			// @yrchen: set initial block label size 
			VertexID vid = (*it)->id;
			if (IsLabelVertex(labels, vid, synthetic)) {
				b_info.label_size = 1;
			} else {
				b_info.label_size = 0; // this init step is necessary
			}

			for (int i = 0; i < nbs.size(); i++)
			{
				b_info.nb_blocks.insert(nbs[i].value);
			}
			b_info_map[color] = b_info;
		}
		else
		{
			blockInfo & blk = b_iter->second;
			blk.size++;
			for (int i = 0; i < nbs.size(); i++)
			{
				blk.nb_blocks.insert(nbs[i].value);
			}
			// @yrchen: add block label size
			VertexID vid = (*it)->id;
			if (IsLabelVertex(labels, vid, synthetic)) {
				blk.label_size += 1;
			}

		}
	}

	// TODO: Merge blocks

	//use shuffle to balance the workload of aggregation on blockInfo to each worker
	vector<BInfoMap>  maps(_num_workers);
	for (BIter it = b_info_map.begin(); it != b_info_map.end(); it++)
	{
		maps[hash_(it->first)][it->first] = it->second;
	}
	b_info_map.clear();

	//shuffle the b_info_map
	all_to_all(maps);

	BInfoMap& mypart = maps[_my_rank];

	for (int i = 0; i < _num_workers; i++)
	{
		if (i != _my_rank)
		{
			BInfoMap& part = maps[i];
			for (BIter it = part.begin(); it != part.end(); it++)
			{
				VertexID color = it->first;
				BIter bit = mypart.find(color);
				if (bit == mypart.end())
				{
					b_info_map[color] = it->second;
				}
				else
				{
					bit->second.size += it->second.size;
					// @yrchen: add label size
					bit->second.label_size += it->second.label_size;

					set<VertexID> & nb_block = it->second.nb_blocks;
					set<VertexID>::iterator setIter;
					for (setIter = nb_block.begin(); setIter != nb_block.end(); setIter++)
					{
						bit->second.nb_blocks.insert(*setIter);
					}
				}
			}
			part.clear();
		}
	}

	//aggregation
	if (_my_rank != MASTER_RANK)
	{
		// @yrchen: For print only -- check label cnt in each block
		int label_cnt = 0;
		int node_cnt = 0;
		for (BIter it = mypart.begin(); it != mypart.end(); it++) {
			label_cnt += it->second.label_size;
			node_cnt += it->second.size;
		}
		cout << "My Part " << _my_rank << " has labels " << label_cnt << " out of " << node_cnt << " nodes." << endl;
		send_data(mypart, MASTER_RANK, 0);
		mypart.clear();
		slave_bcast(blk_to_slv_);
	}
	else
	{
		// @yrchen: For print only -- check label cnt in each block
		int label_cnt = 0;
		int node_cnt = 0;
		for (BIter it = mypart.begin(); it != mypart.end(); it++) {
			label_cnt += it->second.label_size;
			node_cnt += it->second.size;
		}
		cout << "My Part " << _my_rank << " has labels " << label_cnt << " out of " << node_cnt << " nodes." << endl;

		for (int i = 0; i < _num_workers; i++)
		{
			if (i != MASTER_RANK)
			{
				obinstream um = recv_obinstream(MPI_ANY_SOURCE, 0);
				BInfoMap part;
				um >> part;
				for (BIter it = part.begin(); it != part.end(); it++)
				{
					mypart[it->first] = it->second;
				}
				part.clear();
			}
		}

		//-----------------------------
		//%%% for print only %%%
		hash_map<int, int> histogram;
		hash_map<int, int> label_histogram;
		vector<int> zero_nb;
		int zero_nb_size = 0;
        int total_blk = 0;
        int small_blk_size = 0;
        int small_blk_nb_size = 0;
		for (BIter it = mypart.begin(); it != mypart.end(); it++)
		{
			//{ %%% for print only %%%
			int key = num_digits(it->second.size);
			if (it->second.nb_blocks.size() <= 0) {
				zero_nb.push_back((int)it->second.color);
				zero_nb_size += it->second.size;
			} 
            if (it->second.size < 10) {
                total_blk ++;
                small_blk_size += it->second.size;
                small_blk_nb_size += it->second.nb_blocks.size();
            }
			hash_map<int, int>::iterator hit = histogram.find(key);
			if (hit == histogram.end())
				histogram[key] = 1;
			else
				hit->second++;
			//%%% for print only %%% }
			int label_key = num_digits(it->second.label_size);
			hash_map<int, int>::iterator label_hit = label_histogram.find(label_key);
			if (label_hit == label_histogram.end())
				label_histogram[label_key] = 1;
			else
				label_hit->second++;			
		}

		//------------------- report begin -------------------
		cout << "* block size histogram:" << endl;
		for (hash_map<int, int>::iterator hit = histogram.begin(); hit != histogram.end(); hit++)
		{
			cout << "|V|<" << hit->first << ": " << hit->second << " blocks" << endl;
		}
		cout << "|V_nb <= 0| = " << zero_nb.size() << ", its |V| = " << zero_nb_size << endl;
        cout << "|V| < 10, blk # " << total_blk << ", avg blk_size " << (float) small_blk_size / total_blk << ", avg blk nb size " << (float) small_blk_nb_size / total_blk << endl;
		//------------------- report end ---------------------

		//------------------- report begin -------------------
		cout << "* block label size histogram:" << endl;
		for (hash_map<int, int>::iterator hit = label_histogram.begin(); hit != label_histogram.end(); hit++)
		{
			cout << "|label|<" << hit->first << ": " << hit->second << " blocks" << endl;
		}
		//------------------- report end ---------------------

		//convert map<color, blockInfo> to vector
		vector<blockInfo> blocks_info;
		for (BIter bit = mypart.begin(); bit != mypart.end(); bit++)
		{
			blocks_info.push_back(bit->second);
		}
		b_info_map.clear();
        
		// @yrchen: Apply MergeBlock strategy
        /*
        cout << "Merge block..." << endl;
		vector<blockInfo> merge_blk_info;
        auto id_merge_map = MergeBlock(blocks_info, merge_blk_info);
		hash_map<VertexID, int> merge_blk_to_slv;
		partition(merge_blk_info, merge_blk_to_slv);
		UnMergeBlocks(blocks_info, merge_blk_to_slv, id_merge_map, blk_to_slv_);
        */
		partition(blocks_info, blk_to_slv_);   //user will set blk2slv

		//%%%%%% scattering blk2slv
		master_bcast(blk_to_slv_);
	}
}


template<class BDGPartVertexT>
void BDGPartitioner<BDGPartVertexT>::partition(vector<blockInfo> & blocks_info, hash_map<VertexID, int> & blk_to_slv) //Implement the partition strategy base on blockInfo
{
	//Streaming Graph Partitioning for Large Distributed Graphs KDD 2013
	//strategy 4
	//blk2slv : set block B --> worker W

	double eps = 0.1;  //the ratio to adjust the capacity of bins
	int total_count = 0;
	// @yrchen: record blocks_info hashmap index 
	// Warning: block ID (color) might exceed blocks_info size, hence requiring hashmap to store
	hash_map<VertexID, int> block_idx;
	for (int i = 0; i < blocks_info.size(); i++)
	{
		total_count += blocks_info[i].size;
		// @yrchen: record blocks_info hashmap index
		block_idx[blocks_info[i].color] = i;
	}

	// @yrchen: calculate the label size for each block
	int total_label_count = 0;
	for (int i = 0; i < blocks_info.size(); i++)
	{
		total_label_count += blocks_info[i].label_size;
	}
	int label_capacity = 1 * (total_label_count / get_num_workers());
	float label_coe = 1; // factor of label capacity constraint in assignment function
	float capacity_coe = 0.5; // factor of label capacity constraint in assignment function
	float second_neighbor_coe = 1;
	cout << "Total label size is " << total_label_count << ", label_capacity is " << label_capacity << endl;
	cout << "Node capacity coefficient " << capacity_coe << ", label capacity coefficient " << label_coe << endl;
	//set the capacity of bin to the (1+eps)* [avg of number of v in each worker]
	int capacity = (1 + eps) * (total_count / get_num_workers());

	// sort in non-increasing order of block size
	sort(blocks_info.begin(), blocks_info.end());

	int* assigned = new int[_num_workers];
	int* bcount = new int[_num_workers]; //%%% for print only %%%
	// @yrchen: assigned labels in each partitions
	int* label_assigned = new int[_num_workers];
	hash_map<VertexID, int> * countmap = new  hash_map<VertexID, int>[_num_workers]; //store the sum of the size of adjacent blocks in each worker
	//hash_map<VertexID, atomic_ptr_t> * second_countmap = new  hash_map<VertexID, atomic_ptr_t>[_num_workers]; //store the sum of the size of adjacent blocks in each worker
    hash_map<VertexID, int> * second_countmap = new  hash_map<VertexID, int>[_num_workers];
	//vector<atomic<int>> second_countmap_value();
	// =================Parallel Optimization================
	reset_timer(6);
	for (int i = 0; i < blocks_info.size(); i++) {
		auto & block = blocks_info[i];
		block.nb_blocks_vec.resize(block.nb_blocks.size());
		copy(block.nb_blocks.begin(), block.nb_blocks.end(), block.nb_blocks_vec.begin());
		auto color = block.color;
		for (int j = 0; j < _num_workers; j++) {
			//second_countmap[j][color] = atomic_ptr_t(new atomic<int>(0));
            second_countmap[j][color] = 0;
		}
	}
	stop_timer(6);
	cout << "Parallel optimization for " << blocks_info.size() << " costs " << get_timer(6) << " seconds." << endl;
	// =================Parallel Optimization End================

	for (int i = 0; i < _num_workers; i++)
	{
		assigned[i] = 0;
		bcount[i] = 0;
		// @yrchen: assigned labels in each partitions
		label_assigned[i] = 0;
	}

	//allocate each block to workers
	//strategy:
	//block B is always allocated to worker Wi with the highest priority
	// weight : wi = 1 - assigned[Wi] / capacity
	// Ci : the size of one adjacent block of B in worker Wi
	// Si : sum { Ci }
	//priority = Si * wi

	// 1. @yrchen: Lack regularization factor for Si and Wi?
	// priority = Si * (wi)^beta
	// 2. @yrchen: Add label load balancing term Li
	// Li = 1 - label_assigned[Wi] / capacity
	// priority = Si * (wi)^beta * (Li)^gamma
	hash_map<VertexID, int>::iterator cmIter;
    hash_map<VertexID, int>::iterator second_cmIter;
	//hash_map<VertexID, atomic_ptr_t>::iterator second_cmIter;
	// @yrchen: debugging variables
	double assign_time = 0;
	double update_first_time = 0;
	double update_second_time = 0;
	int first_hop_neighbor_num = 0;
	unsigned long int second_hop_neighbor_num = 0;
    int output_step = (int) blocks_info.size() / 30;
    //output_step = 10;
    cout << "Start block assignment..." << endl;
	for (int i = 0; i < blocks_info.size(); i++)
	{
        // cout << "Assign block " << blocks_info[i].color << endl;
		blockInfo & cur = blocks_info[i];
		double max = 0;
		int wid = 0;
		bool is_allocated = false;
		ResetTimer(5);
		for (int j = 0; j < _num_workers; j++)
		{
			double priority = 0;
			// @yrchen: add second-hop neighbor searching
			second_cmIter = second_countmap[j].find(cur.color);
			int second_neighbor = 0;
			second_neighbor_coe = 1;
			if (second_cmIter != second_countmap[j].end()) {
				//second_neighbor += second_cmIter->second->load();
                second_neighbor += second_cmIter->second;
				// TODO: add this term into priority
				//second_neighbor_coe = 15 / cur.nb_blocks.size();
			}

			cmIter = countmap[j].find(cur.color);
			if (cmIter != countmap[j].end())
			{
				// priority = cmIter->second * (1 - assigned[j] / capacity);  //calculate the priority of each work for current block
				// priority = cmIter->second * pow((1 - assigned[j] / capacity), capacity_coe);
				priority = (cmIter->second + second_neighbor_coe * second_neighbor) * pow((1 - assigned[j] / capacity), capacity_coe);
				// @yrchen: add label capacity into priority scores
				//priority *= pow((1 - (label_assigned[j] + cur.label_size) / label_capacity), label_coe);
			} else if ( assigned[j] < 0.01 * capacity) { // for cold start, prevent from worst case -- always assigned to one machine 
				priority = 1;
			}

			// @yrchen: This ensures strong load balancing limit. But do we need such strong limit?
			if ((priority > max) && (assigned[j] + cur.size <= capacity))
			{
				max = priority;
				wid = j;
				is_allocated = true;
			}
		}
		// allocation is failed, current block should be allocated to the work has much available space
		if (!is_allocated)
		{
			int minSize = assigned[0];
			for (int j = 1; j < _num_workers; j++)
			{
				if (minSize > assigned[j])
				{
					minSize = assigned[j];
					wid = j;
				}
			}
		}
		// @yrchen: record time
		StopTimer(5);
		assign_time += get_timer(5);
		reset_timer(5);

		blk_to_slv[cur.color] = wid;
		assigned[wid] += cur.size;
		bcount[wid] ++;
        //cout << "Block " << cur.color << " assignied to part " << wid << endl;
		// @yrchen: assigned labels in each partitions
		label_assigned[wid] += cur.label_size;
		//update the countmap in worker W, insert or update the adjacent blocks' values with current block's size
        
		for (set<int>::iterator it = cur.nb_blocks.begin(); it != cur.nb_blocks.end(); it++)
		{
			cmIter = countmap[wid].find(*it);
			if (cmIter != countmap[wid].end())
				cmIter->second += cur.size;
			else
				countmap[wid][*it] = cur.size;
		}
		// @yrchen: record time
		StopTimer(5);
		update_first_time += get_timer(5);
		reset_timer(5);

		// @yrchen: update the second_countmap in worker W, insert or update the adjacent blocks' values with current block's size
		first_hop_neighbor_num += cur.nb_blocks.size();

		// for (set<int>::iterator it = cur.nb_blocks.begin(); it != cur.nb_blocks.end(); it++) {
		// 	// find the neighbors of this neighbor block
		// 	int neighbor_color = block_idx[*it];
		// 	blockInfo & neighbor_block = blocks_info[neighbor_color];
		// 	second_hop_neighbor_num += neighbor_block.nb_blocks.size();
		// 	for (set<int>::iterator nit = neighbor_block.nb_blocks.begin(); nit != neighbor_block.nb_blocks.end(); nit++) {
		// 		second_cmIter = second_countmap[wid].find(*nit);
		// 		if (second_cmIter != second_countmap[wid].end())
		// 			second_cmIter->second += cur.size;
		// 		else
		// 			second_countmap[wid][*nit] = cur.size;
		// 	}		
		// }

		// @yrchen: Parallel impl to accelerate
//#pragma	omp parallel for
        // cout << "Finish update 1-hop nb. Start update 2-hop nb " << cur.nb_blocks_vec.size() << "." << endl;
        /*
		for (size_t k = 0; k < cur.nb_blocks_vec.size(); k++) {
			// find the neighbors of this neighbor block
			int color_idx = cur.nb_blocks_vec[k];
			int neighbor_color = block_idx[color_idx];
			blockInfo & neighbor_block = blocks_info[neighbor_color];
			second_hop_neighbor_num += neighbor_block.nb_blocks_vec.size();
            // cout << "Update 2-hop nb for block " << cur.color << ", 1-hop nb " << neighbor_block.color << ", second_countmap size " << second_countmap[wid].size() << endl;
			for (size_t l = 0; l < neighbor_block.nb_blocks_vec.size(); l++) {
                //cout << "2-hop nb:" << neighbor_block.nb_blocks_vec[l] << endl;
				int second_color = neighbor_block.nb_blocks_vec[l];
				second_cmIter = second_countmap[wid].find(second_color);
				if (second_cmIter != second_countmap[wid].end()) {
					//second_countmap[wid][second_color]->fetch_add(cur.size);
					second_countmap[wid][second_color] += cur.size; 
				} else {
					//cout << "Not found error for color " << second_color << endl;
					second_countmap[wid][second_color] = cur.size; 
				}
				// second_countmap[wid][second_color]->fetch_add(cur.size);
				//second_countmap[wid][second_color] += cur.size; 
			}		

		}*/

		// @yrchen: record time
		StopTimer(5);
		update_second_time += get_timer(5);
		reset_timer(5);
		if ((i % output_step == 0 && i != 0) || (i <= 20)) {
			cout << "=========assigned block " << i << " | assigned time " << assign_time << "s | update first hop " 
			<< update_first_time << "s | update second hop " << update_second_time << "s | avg 1-hop neighbor " 
			<< first_hop_neighbor_num / (i+1) << " | avg 2-hop neighbor " << second_hop_neighbor_num / (i+1) << "=============" << endl;
		}
	}


	cout << "* per-machine block assignment:" << endl;
	for (int i = 0; i < _num_workers; i++)
	{
		// cout << "Machine_" << i << " is assigned " << bcount[i] << " blocks, " << assigned[i] << " vertices" << endl;
		// @yrchen: add label assignment output
		cout << "Machine_" << i << " is assigned " << bcount[i] << " blocks, " << assigned[i] << " vertices, " << label_assigned[i] << " labels." << endl;
	}

	cout << "=======Evaluating partition quality/locality=======" << endl;
	for (int i = 0; i < blocks_info.size(); i ++) {
		auto block = blocks_info[i];
		int color = block.color;
		int wid = blk_to_slv[color];

	}

	for (int i = 0; i < _num_workers; i++) {


	}

	delete[] bcount;
	delete[] assigned;
	delete[] countmap;
	// @yrchen: delete countmap
	delete[] second_countmap;
}


//=====================================normal_BDGPartitioner=====================================
normal_BDGPartValue::normal_BDGPartValue() : color(-1) { }

//no vertex-add, will not be called
ibinstream& operator << (ibinstream& m, const normal_BDGPartValue& v)
{
	m << v.color;
	m << v.neighbors;
	m << v.nbs_info;
	return m;
}
obinstream& operator >> (obinstream& m, normal_BDGPartValue& v)
{
	m >> v.color;
	m >> v.neighbors;
	m >> v.nbs_info;
	return m;
}

void normal_BDGPartVorCombiner::combine(VertexID& old, const VertexID& new_msg) { } //ignore new_msg

void normal_BDGPartHashMinCombiner::combine(VertexID& old, const VertexID& new_msg)
{
	if (old > new_msg)
		old = new_msg;
}

normal_BDGPartVertex::normal_BDGPartVertex()
{
	srand((unsigned)time(NULL));
}

void normal_BDGPartVertex::broadcast(VertexID msg)
{
	vector<VertexID>& nbs = value().neighbors;
	for (int i = 0; i < nbs.size(); i++)
	{
		send_message(nbs[i], msg);
	}
}

void normal_BDGPartVertex::compute(MessageContainer& messages) //Voronoi Diagram partitioning algorithm
{
	if (step_num() == 1)
	{
		double samp = ((double)rand()) / RAND_MAX;
		if (samp <= global_sampling_rate)   //sampled
		{
			value().color = id;
			broadcast(id);
		}
		else   //not sampled
		{
			value().color = -1; //-1 means not assigned color
		}
		vote_to_halt();
	}
	else if (step_num() >= global_max_hop)
		vote_to_halt();
	else
	{
		if (value().color == -1)
		{
			VertexID msg = *(messages.begin());
			value().color = msg;
			broadcast(msg);
		}
		vote_to_halt();
	}
}


void normal_BDGPartitioner::set_voronoi_combiner()
{
	combiner_ = new normal_BDGPartVorCombiner;
	global_combiner = combiner_;
}

void normal_BDGPartitioner::set_hashmin_combiner()
{
	combiner_ = new normal_BDGPartHashMinCombiner;
	global_combiner = combiner_;
}

normal_BDGPartVertex* normal_BDGPartitioner::to_vertex(char* line)
{
	try {
		char* pch;
		pch = strtok(line, "\t");
		normal_BDGPartVertex* v = new normal_BDGPartVertex;
		v->id = atoi(pch);
		pch = strtok(NULL, " ");
		int num = atoi(pch);
		//v->value().color=-1;//default is -1
		while (num--)
		{
			pch = strtok(NULL, " ");
			v->value().neighbors.push_back(atoi(pch));
		}
		return v;
	} catch (const std::exception& exc) {
		std::cerr << exc.what() << "\n";
		return NULL;
	}
	
}

void normal_BDGPartitioner::to_line(normal_BDGPartVertex* v, BufferedWriter& writer) //key: "vertexID blockID slaveID"
{
	sprintf(buf_, "%d %d\t", v->id, _my_rank);
	writer.write(buf_);
	vector<kvpair>& vec = v->value().nbs_info;
	for (int i = 0; i < vec.size(); i++)
	{
		sprintf(buf_, "%d %d ", vec[i].vid, vec[i].value);
		writer.write(buf_);
	}
	writer.write("\n");
}

void normal_BDGPartitioner::nb_info_exchange()
{
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 1 (send hash_table) =============" << endl;
	MapVec maps(_num_workers);
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		VertexID vid = (*it)->id;
		int blockID = (*it)->value().color;
		kvpair trip = { vid, blockID };
		vector<VertexID>& nbs = (*it)->value().neighbors;
		for (int i = 0; i < nbs.size(); i++)
		{
			maps[hash_(nbs[i])][vid] = trip;
		}
	}

	ExchangeT recvBuf(_num_workers);
	all_to_all(maps, recvBuf);
	hash_map<VertexID, kvpair>& mymap = maps[_my_rank];
	// gather all table entries
	for (int i = 0; i < _num_workers; i++)
	{
		if (i != _my_rank)
		{
			maps[i].clear(); //free sent table
			vector<IDTrip>& entries = recvBuf[i];
			for (int j = 0; j < entries.size(); j++)
			{
				IDTrip& idm = entries[j];
				mymap[idm.id] = idm.trip;
			}
		}
	}

	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 2 =============" << endl;
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		normal_BDGPartVertex& vcur = **it;
		vector<VertexID>& nbs = vcur.value().neighbors;
		vector<kvpair>& infos = vcur.value().nbs_info;
		for (int i = 0; i < nbs.size(); i++)
		{
			VertexID nb = nbs[i];
			kvpair trip = mymap[nb];
			infos.push_back(trip);
		}
	}
	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
}

//=====================================normal_BDGPartitioner=====================================


//=====================================label_BDGPartitioner======================================
label_BDGPartValue::label_BDGPartValue() : color(-1) { }

//no vertex-add, will not be called
ibinstream& operator << (ibinstream& m, const label_BDGPartValue& v)
{
	m << v.color;
	m << v.label;
	m << v.neighbors;
	m << v.nbs_info;
	return m;
}
obinstream& operator >> (obinstream& m, label_BDGPartValue& v)
{
	m >> v.color;
	m >> v.label;
	m >> v.neighbors;
	m >> v.nbs_info;
	return m;
}

void label_BDGPartVorCombiner::combine(VertexID& old, const VertexID& new_msg) { } //ignore new_msg

void label_BDGPartHashMinCombiner::combine(VertexID& old, const VertexID& new_msg)
{
	if (old > new_msg)
		old = new_msg;
}

label_BDGPartVertex::label_BDGPartVertex()
{
	srand((unsigned)time(NULL));
}

void label_BDGPartVertex::broadcast(VertexID msg)
{
	vector<nbInfo>& nbs = value().neighbors;
	for (int i = 0; i < nbs.size(); i++)
	{
		send_message(nbs[i].vid, msg);
	}
}

void label_BDGPartVertex::compute(MessageContainer& messages) //Voronoi Diagram partitioning algorithm
{
	if (step_num() == 1)
	{
		double samp = ((double)rand()) / RAND_MAX;
		if (samp <= global_sampling_rate)   //sampled
		{
			value().color = id;
			broadcast(id);
		}
		else   //not sampled
		{
			value().color = -1; //-1 means not assigned color
		}
		vote_to_halt();
	}
	else if (step_num() >= global_max_hop)
		vote_to_halt();
	else
	{
		if (value().color == -1)
		{
			VertexID msg = *(messages.begin());
			value().color = msg;
			broadcast(msg);
		}
		vote_to_halt();
	}
}


void label_BDGPartitioner::set_voronoi_combiner()
{
	combiner_ = new label_BDGPartVorCombiner;
	global_combiner = combiner_;
}

void label_BDGPartitioner::set_hashmin_combiner()
{
	combiner_ = new label_BDGPartHashMinCombiner;
	global_combiner = combiner_;
}

label_BDGPartVertex* label_BDGPartitioner::to_vertex(char* line)
{
	label_BDGPartVertex* v = new label_BDGPartVertex;
	char* pch;
	pch = strtok(line, " ");
	v->id = atoi(pch);
	pch = strtok(NULL, "\t");
	v->value().label = *pch;
	while ((pch = strtok(NULL, " ")) != NULL)
	{
		nbInfo item;
		item.vid = atoi(pch);
		pch = strtok(NULL, " ");
		item.label = *pch;
		v->value().neighbors.push_back(item);
	}
	return v;
}

void label_BDGPartitioner::to_line(label_BDGPartVertex* v, BufferedWriter& writer) //key: "vertexID blockID slaveID"
{
	sprintf(buf_, "%d %c %d\t", v->id, v->value().label, _my_rank);
	writer.write(buf_);

	vector<nbInfo>& nbs = v->value().neighbors;
	vector<kvpair>& infos = v->value().nbs_info;

	for (int i = 0; i < nbs.size(); i++)
	{
		sprintf(buf_, "%d %c %d ", nbs[i].vid, nbs[i].label, infos[i].value);
		writer.write(buf_);
	}
	writer.write("\n");
}

void label_BDGPartitioner::nb_info_exchange()
{
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 1 (send hash_table) =============" << endl;
	MapVec maps(_num_workers);
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		VertexID vid = (*it)->id;
		int blockID = (*it)->value().color;
		kvpair trip = { vid, blockID };
		vector<nbInfo>& nbs = (*it)->value().neighbors;
		for (int i = 0; i < nbs.size(); i++)
		{
			maps[hash_(nbs[i].vid)][vid] = trip;
		}
	}

	ExchangeT recvBuf(_num_workers);
	all_to_all(maps, recvBuf);
	hash_map<VertexID, kvpair>& mymap = maps[_my_rank];
	// gather all table entries
	for (int i = 0; i < _num_workers; i++)
	{
		if (i != _my_rank)
		{
			maps[i].clear(); //free sent table
			vector<IDTrip>& entries = recvBuf[i];
			for (int j = 0; j < entries.size(); j++)
			{
				IDTrip& idm = entries[j];
				mymap[idm.id] = idm.trip;
			}
		}
	}

	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 2 =============" << endl;
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		label_BDGPartVertex& vcur = **it;
		vector<nbInfo>& nbs = vcur.value().neighbors;
		vector<kvpair>& infos = vcur.value().nbs_info;
		for (int i = 0; i < nbs.size(); i++)
		{
			nbInfo & nb = nbs[i];
			kvpair trip = mymap[nb.vid];
			infos.push_back(trip);
		}
	}
	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
}

//=====================================label_BDGPartitioner======================================


//======================================attr_BDGPartitioner======================================
attr_BDGPartValue::attr_BDGPartValue() : color(-1) { }

//no vertex-add, will not be called
ibinstream& operator << (ibinstream& m, const attr_BDGPartValue& v)
{
	m << v.color;
	m << v.attr;
	m << v.neighbors;
	m << v.nbs_info;
	return m;
}
obinstream& operator >> (obinstream& m, attr_BDGPartValue& v)
{
	m >> v.color;
	m >> v.attr;
	m >> v.neighbors;
	m >> v.nbs_info;
	return m;
}

void attr_BDGPartVorCombiner::combine(VertexID& old, const VertexID& new_msg)	{ } //ignore new_msg

void attr_BDGPartHashMinCombiner::combine(VertexID& old, const VertexID& new_msg)
{
	if (old > new_msg)
		old = new_msg;
}

attr_BDGPartVertex::attr_BDGPartVertex()
{
	srand((unsigned)time(NULL));
}

void attr_BDGPartVertex::broadcast(VertexID msg)
{
	vector<nbAttrInfo<AttrValueT> >& nbs = value().neighbors;
	for (int i = 0; i < nbs.size(); i++)
	{
		send_message(nbs[i].vid, msg);
	}
}

void attr_BDGPartVertex::compute(MessageContainer& messages) //Voronoi Diagram partitioning algorithm
{
	if (step_num() == 1)
	{
		double samp = ((double)rand()) / RAND_MAX;
		if (samp <= global_sampling_rate)   //sampled
		{
			value().color = id;
			broadcast(id);
		}
		else   //not sampled
		{
			value().color = -1; //-1 means not assigned color
		}
		vote_to_halt();
	}
	else if (step_num() >= global_max_hop)
		vote_to_halt();
	else
	{
		if (value().color == -1)
		{
			VertexID msg = *(messages.begin());
			value().color = msg;
			broadcast(msg);
		}
		vote_to_halt();
	}
}


void attr_BDGPartitioner::set_voronoi_combiner()
{
	combiner_ = new attr_BDGPartVorCombiner;
	global_combiner = combiner_;
}

void attr_BDGPartitioner::set_hashmin_combiner()
{
	combiner_ = new attr_BDGPartHashMinCombiner;
	global_combiner = combiner_;
}

attr_BDGPartVertex* attr_BDGPartitioner::to_vertex(char* line)
{
	//char sampleLine[] = "0\t1983 166 1 ball;tennis \t3 1 (0.1 )2 (0.2 )3 (0.3 )";
	attr_BDGPartVertex* v = new attr_BDGPartVertex;
	char* pch;
	pch = strtok(line, "\t"); //0\t
	v->id = atoi(pch);

	vector<AttrValueT> a;
	pch = strtok(NULL, "\t"); //1983 166 ball;tennis \t
	char* attr_line = new char[strlen(pch) + 1];
	strcpy(attr_line, pch);
	attr_line[strlen(pch)] = '\0';

	pch = strtok(NULL, " ");
	int num = atoi(pch);
	while (num--)
	{
		nbAttrInfo<AttrValueT> item;
		pch = strtok(NULL, " ");
		item.vid = atoi(pch);
		if(USE_MULTI_ATTR)
		{
			pch = strtok(NULL, " ");
			item.attr.get_attr_vec().push_back(pch);
		}
		v->value().neighbors.push_back(item);
	}

	pch = strtok(attr_line, " ");
	a.push_back(pch);
	while ((pch = strtok(NULL, " ")) != NULL)
		a.push_back(pch);
	v->value().attr.init_attr_vec(a);

	delete[] attr_line;
	return v;
}

void attr_BDGPartitioner::to_line(attr_BDGPartVertex* v, BufferedWriter& writer) //key: "vertexID blockID slaveID"
{
	//char sampleOutputLine[] = "0 5\t1983 166 1 ball;tennis \t1 (0.1 )1 2 (0.2 )2 ";
	sprintf(buf_, "%d %d\t", v->id, _my_rank);
	writer.write(buf_);

	vector<AttrValueT>& attrVec = v->value().attr.get_attr_vec();
	for (size_t i = 0; i < attrVec.size(); ++i)
	{
		sprintf(buf_, "%s ", attrVec[i].c_str());
		writer.write(buf_);
	}
	writer.write("\t");

	vector<nbAttrInfo<AttrValueT> >& nbs = v->value().neighbors;
	vector<kvpair>& infos = v->value().nbs_info;

	for (int i = 0; i < infos.size(); i++)
	{
		if(!USE_MULTI_ATTR)
			sprintf(buf_, "%d %d ", infos[i].vid, infos[i].value);
		else
			sprintf(buf_, "%d %s %d ", nbs[i].vid, nbs[i].attr.get_attr_vec().front().c_str(), infos[i].value);
		writer.write(buf_);
	}
	writer.write("\n");
}

void attr_BDGPartitioner::nb_info_exchange()
{
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 1 (send hash_table) =============" << endl;
	MapVec maps(_num_workers);
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		VertexID vid = (*it)->id;
		int blockID = (*it)->value().color;
		kvpair trip = { vid, blockID };
		vector<nbAttrInfo<AttrValueT> >& nbs = (*it)->value().neighbors;
		for (int i = 0; i < nbs.size(); i++)
		{
			maps[hash_(nbs[i].vid)][vid] = trip;
		}
	}

	ExchangeT recvBuf(_num_workers);
	all_to_all(maps, recvBuf);
	hash_map<VertexID, kvpair>& mymap = maps[_my_rank];
	// gather all table entries
	for (int i = 0; i < _num_workers; i++)
	{
		if (i != _my_rank)
		{
			maps[i].clear(); //free sent table
			vector<IDTrip>& entries = recvBuf[i];
			for (int j = 0; j < entries.size(); j++)
			{
				IDTrip& idm = entries[j];
				mymap[idm.id] = idm.trip;
			}
		}
	}

	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
	ResetTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << "============= Neighbor InfoExchange Phase 2 =============" << endl;
	for (VertexIter it = vertexes_.begin(); it != vertexes_.end(); it++)
	{
		attr_BDGPartVertex& vcur = **it;
		vector<nbAttrInfo<AttrValueT> >& nbs = vcur.value().neighbors;
		vector<kvpair>& infos = vcur.value().nbs_info;
		for (int i = 0; i < nbs.size(); i++)
		{
			nbAttrInfo<AttrValueT> & nb = nbs[i];
			kvpair trip = mymap[nb.vid];
			infos.push_back(trip);
		}
	}
	StopTimer(4);
	if (_my_rank == MASTER_RANK)
		cout << get_timer(4) << " seconds elapsed" << endl;
}

//======================================attr_BDGPartitioner======================================
