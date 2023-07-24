#include <dgl/runtime/object.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/packed_func_ext.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <string>
#include "./rpc.h"
#include "../c_api_common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <dmlc/blockingconcurrentqueue.h>
#include "../graph/heterograph.h"
#include "../runtime/cuda/cuda_common.h"
#include "lfu.h"
#include <chrono>
#include "lru.h"
#include <set>
#include <dgl/aten/array_ops.h>



using namespace dgl::runtime;
using namespace dgl::rpc;
using namespace dgl;


#define MAX_PENDING_SAMPLES 5
#define MAX_WORKER 8
#define MAX_NODES 10000000
#define NUM_WORKER_PER_NVLINK 4
#define MAX_NVLINKS 2
#define MAX_LAYERS 5

int worker_num;
int number_of_nodes;
int cache_size_per_gpu;
int feature_dim;
int max_inputs_length;
int samples_per_worker;
NDArray cpu_cache_buffer;
int layers;
int batch_size;
cudaStream_t global_torch_cuda_stream;


//global node map array used in LFU
//dataNode ** gpu_cache_node_map;

//global node map array used in LRU
//CacheNode ** gpu_cache_node_map;


//global node map array used in full and FIFO
NDArray gpu_cache_node_map[MAX_NVLINKS];

// record meta data of each sample
struct SampleMetaData{
	int worker_id;
	int sample_idx;
	int inputs_length;
	int seeds_length;
};
SampleMetaData sample_meta_data[MAX_WORKER][MAX_PENDING_SAMPLES];

NDArray sample_feature_buffers[MAX_WORKER][MAX_PENDING_SAMPLES];


// due to asyn execution in cuda stream, we use this struct to track the state of caching task
// expected_value records the number of task should be fininshed
// finished_value records the number of task has been finished
struct CachingTaskFinishCount{
	std::atomic<int> expected_value;
	std::atomic<int> finished_value;
};

CachingTaskFinishCount caching_task_finish_count[MAX_WORKER][MAX_PENDING_SAMPLES];


// linux fifo for inter-process communication
// recv_fifo: from receiving process to dispatching thread in caching process
// cach_fifo: from working thread in caching process to training process
int recv_fifo_fds[MAX_WORKER];
int cache_fifo_fds[MAX_WORKER];


struct CachingTask{
	int worker_id;
	int sample_idx;
	NDArray node_id;
	NDArray req_idx;
	NDArray return_tensor;
	CachingTask(){}
	CachingTask(int w_id, int s, NDArray nid, NDArray ridx, NDArray ret): worker_id(w_id), sample_idx(s), node_id(nid), req_idx(ridx), return_tensor(ret){
	}
};

// concurrent queue for dispatching thread and working thread
// dispatching thrad enqueues caching task
// working thread dequeues caching task
dmlc::moodycamel::BlockingConcurrentQueue<CachingTask> task_queue[MAX_WORKER];


// read n bytes from fd
void read_nbytes(int fd, void * p, int length){
	char * ptr = static_cast<char*>(p);
	int cnt = length;
	while(cnt){
		int n = read(fd, ptr, cnt);
		cnt -= n;
		ptr += n;
	}
	return;
}

void CUDART_CB check_caching_task_finished(void * data)
{
	//std::cout << "start callback" << std::endl;
	SampleMetaData * meta_data = static_cast<SampleMetaData*>(data);
	int worker_id = meta_data->worker_id;
	int cur_idx = meta_data->sample_idx % MAX_PENDING_SAMPLES;
	CachingTaskFinishCount& ctfc = caching_task_finish_count[worker_id][cur_idx];
	int cur_count = ctfc.finished_value.fetch_add(1,std::memory_order_relaxed)+1;
	if(cur_count == ctfc.expected_value){
		write(cache_fifo_fds[worker_id],  static_cast<void*>(meta_data), sizeof(SampleMetaData));
	}
	//std::cout << "finsihed caching task worker id : " << meta_data->worker_id << " sample_idx: " << meta_data->sample_idx << std::endl;
}



void caching_dispatching_thread(int worker_id)
{
	//std::cout << "start caching dispatching thread " << worker_id << std::endl;

	int recv_fifo_fd = recv_fifo_fds[worker_id];

	// allocate DGL NDArray and create IpcMemHandle of data ptr in NDArray
	// and send cudaIpcMemHandle data to training process
	cudaIpcMemHandle_t cache_cuda_mem_ipc_handles[MAX_PENDING_SAMPLES];
	for(int i = 0; i < MAX_PENDING_SAMPLES; ++i){
		sample_feature_buffers[worker_id][i] = NDArray::Empty({max_inputs_length, feature_dim}, {kDLFloat, 32, 1}, {kDLGPU, worker_id});
		cudaIpcGetMemHandle(cache_cuda_mem_ipc_handles+i, static_cast<void*>(sample_feature_buffers[worker_id][i]->data));
	}
	write(cache_fifo_fds[worker_id], static_cast<void*>(cache_cuda_mem_ipc_handles), MAX_PENDING_SAMPLES * sizeof(cudaIpcMemHandle_t));

	//std::cout << "finish write cuda" << std::endl;
	
	// we should know the number of worker on current nvlink switch
	int worker_num_on_cur_nvlink;
	if(worker_num <= NUM_WORKER_PER_NVLINK)
		worker_num_on_cur_nvlink = worker_num;
	else if(worker_id < NUM_WORKER_PER_NVLINK)
		worker_num_on_cur_nvlink = NUM_WORKER_PER_NVLINK;
	else
		worker_num_on_cur_nvlink = worker_num - NUM_WORKER_PER_NVLINK;
	//std::cout << "in caching dispatching thread " << worker_id << " with worker num on nvlink" << worker_num_on_cur_nvlink << std::endl; 


	for(int s = 0; s < samples_per_worker; ++s){
		int cur_sample_idx = s % MAX_PENDING_SAMPLES;
		SampleMetaData& smd = sample_meta_data[worker_id][cur_sample_idx];


		// read meta info from receiving process
		read_nbytes(recv_fifo_fd, static_cast<void*>(&smd), sizeof(SampleMetaData));
		int inputs_length = smd.inputs_length;

		// get input nodes tensor from shared memory which is created by receiving process
		char  inputs_name[100];
		sprintf(inputs_name, "w%ds%d_inputs", worker_id, s);
		NDArray input_nodes = NDArray::EmptyShared(std::string(inputs_name), {inputs_length}, {kDLInt, 64, 1}, DLContext{kDLCPU,0}, false);
		int64_t * input_nodes_data = static_cast<int64_t*>(input_nodes->data);
		//std::cout << "caching recv samples: " << smd.worker_id << " " << smd.sample_idx << " " << smd.inputs_length << " " << smd.seeds_length << " " << std::endl;


		// split node ides to each caching thread using round robin fashion
		std::vector<std::vector<int64_t>> node_ids_by_gpu(worker_num_on_cur_nvlink);
		std::vector<std::vector<int64_t>> node_req_idx_by_gpu(worker_num_on_cur_nvlink);
		for(int i = 0; i < worker_num_on_cur_nvlink; ++i){
			node_ids_by_gpu[i].reserve(inputs_length);
			node_req_idx_by_gpu[i].reserve(inputs_length);
		}
		for(int64_t i = 0; i < inputs_length; ++i){
			int64_t cur_node_id = input_nodes_data[i];
			int64_t target_gpu_id = cur_node_id % worker_num_on_cur_nvlink;
			node_ids_by_gpu[target_gpu_id].push_back(cur_node_id);
			node_req_idx_by_gpu[target_gpu_id].push_back(i);
		}

		// create a temporary buffere for return feature tensor of each sample
		NDArray sample_return_feature = sample_feature_buffers[worker_id][cur_sample_idx].CreateView({inputs_length, feature_dim}, {kDLFloat, 32, 1});

		// to send what to each queue?
		// need to gather which feature: global node id
		// need to know where to get adta
		// need to know which column should be put
		// need to know how to know whether 
		// QUESTION: HOW TO SET expected value?

		CachingTaskFinishCount & ctfc = caching_task_finish_count[worker_id][cur_sample_idx];
		ctfc.expected_value = 0;
		ctfc.finished_value = 0;
		for(int i = 0; i < worker_num_on_cur_nvlink; ++i){
			// before enqueue caching task, we add the expected value for each caching task
			ctfc.expected_value.fetch_add(1, std::memory_order_relaxed);

			// we enqueu caching task by consider nvlink topology
			if(worker_id < NUM_WORKER_PER_NVLINK)
				task_queue[i].enqueue(CachingTask(worker_id, s, NDArray::FromVector(node_ids_by_gpu[i]), NDArray::FromVector(node_req_idx_by_gpu[i]), sample_return_feature));
			else
				task_queue[i+NUM_WORKER_PER_NVLINK].enqueue(CachingTask(worker_id, s, NDArray::FromVector(node_ids_by_gpu[i]), NDArray::FromVector(node_req_idx_by_gpu[i]), sample_return_feature));
				
		}
		//std::cout << "dispatching thread finished enqueue" << smd.worker_id << " " << smd.sample_idx << std::endl;
		//std::this_thread::sleep_for(std::chrono::seconds(2));
	}
	


}

void caching_working_thread(int target_gpu_id)
{

	//std::cout << "start caching working thread " << target_gpu_id << std::endl;
	//
	bool gpu_cache_is_full = false;
	int64_t gpu_cache_cur_idx = 0;
	int target_nvlink_id = target_gpu_id / NUM_WORKER_PER_NVLINK;

	int worker_num_on_cur_nvlink;
	if(worker_num <= NUM_WORKER_PER_NVLINK)
		worker_num_on_cur_nvlink = worker_num;
	else if(target_gpu_id < NUM_WORKER_PER_NVLINK)
		worker_num_on_cur_nvlink = NUM_WORKER_PER_NVLINK;
	else
		worker_num_on_cur_nvlink = worker_num - NUM_WORKER_PER_NVLINK;

	// data structure for dynamic caching 
	//LRU lru_cache(gpu_cache_node_map);
	//LFU lfu_cache(gpu_cache_node_map);

	// data structure for fifo caching
	NDArray gpu_cache_reverse_idx = NDArray::Empty({cache_size_per_gpu}, {kDLInt, 64, 1}, {kDLCPU, 0});

	// data structure for static caching
	// we should use gpu_cache_node_map on target nvlink 
	int64_t * gpu_cache_node_map_data = static_cast<int64_t*>(gpu_cache_node_map[target_nvlink_id]->data);
	int64_t * gpu_cache_reverse_idx_data = static_cast<int64_t*>(gpu_cache_reverse_idx->data);

	// gpu_cache_buffer on this target gpu 
	NDArray gpu_cache_buffer = NDArray::Empty({cache_size_per_gpu, feature_dim}, {kDLFloat, 32, 1}, {kDLGPU, target_gpu_id});

	// a dedicted CUDA stream for data copy and kernel execution
	DGLStreamHandle target_gpu_stream;
	DGLStreamCreate(kDLGPU, target_gpu_id, &target_gpu_stream);
	DGLSetStream(kDLGPU, target_gpu_id, target_gpu_stream);

	// enable P2P communication from target gpu to other gpu
	cudaSetDevice(target_gpu_id);
	for(int gpu_id = 0; gpu_id < worker_num; ++ gpu_id){
		if(gpu_id == target_gpu_id) continue;
		cudaDeviceEnablePeerAccess(gpu_id, 0);
	}

	// create some temporary NDArray
	NDArray cached_idx_on_cpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLCPU, 0});
	NDArray cached_idx_on_gpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLGPU, target_gpu_id});
	NDArray cached_req_idx_on_cpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLCPU, 0});
	NDArray cached_req_idx_on_gpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLGPU, target_gpu_id});
	NDArray not_cached_nid_on_cpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLCPU, 0});
	NDArray not_cached_new_idx_on_cpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLCPU, 0});
	NDArray not_cached_new_idx_on_gpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLGPU, target_gpu_id});
	NDArray not_cached_req_idx_on_cpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLCPU, 0});
	NDArray not_cached_req_idx_on_gpu = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLGPU, target_gpu_id});
	NDArray not_cached_feature_on_cpu = NDArray::Empty({max_inputs_length, feature_dim}, {kDLFloat, 32, 1}, {kDLCPU, 0});
	NDArray not_cached_feature_on_gpu = NDArray::Empty({max_inputs_length, feature_dim}, {kDLFloat, 32, 1}, {kDLGPU, target_gpu_id});

	// pinned each NDArray memory
	// used for async copy in CUDA stream
	int64_t * cached_idx_on_cpu_data = static_cast<int64_t*>(cached_idx_on_cpu->data);
	int64_t * cached_req_idx_on_cpu_data = static_cast<int64_t*>(cached_req_idx_on_cpu->data);
	int64_t * not_cached_nid_on_cpu_data = static_cast<int64_t*>(not_cached_nid_on_cpu->data);
	int64_t * not_cached_new_idx_on_cpu_data = static_cast<int64_t*>(not_cached_new_idx_on_cpu->data);
	int64_t * not_cached_req_idx_on_cpu_data = static_cast<int64_t*>(not_cached_req_idx_on_cpu->data);
	float * not_cached_feature_on_cpu_data = static_cast<float*>(not_cached_feature_on_cpu->data);
	cudaHostRegister(static_cast<void*>(cached_idx_on_cpu_data), cached_idx_on_cpu.GetSize(), cudaHostRegisterDefault);
	cudaHostRegister(static_cast<void*>(cached_req_idx_on_cpu_data), cached_req_idx_on_cpu.GetSize(), cudaHostRegisterDefault);
	cudaHostRegister(static_cast<void*>(not_cached_new_idx_on_cpu_data), not_cached_new_idx_on_cpu.GetSize(), cudaHostRegisterDefault);
	cudaHostRegister(static_cast<void*>(not_cached_req_idx_on_cpu_data), not_cached_req_idx_on_cpu.GetSize(), cudaHostRegisterDefault);
	cudaHostRegister(static_cast<void*>(not_cached_feature_on_cpu_data), not_cached_feature_on_cpu.GetSize(), cudaHostRegisterDefault);


	//std::cout << "worker_num: " << worker_num << " samples_per_worker " << samples_per_worker << std::endl;
	for(int ct_n = 0; ct_n < worker_num_on_cur_nvlink * samples_per_worker; ++ct_n){
		CachingTask ct;
		task_queue[target_gpu_id].wait_dequeue(ct);

		int worker_id = ct.worker_id;
		int sample_idx = ct.sample_idx;
		//std::cout << "caching working thread " << target_gpu_id << " recv " << worker_id << " " << sample_idx << std::endl;
		NDArray ret = ct.return_tensor;

		int64_t len = ct.node_id->shape[0];
		int64_t* node_id_data = static_cast<int64_t*>(ct.node_id->data);
		int64_t* req_idx_data = static_cast<int64_t*>(ct.req_idx->data);

		auto access_start = std::chrono::high_resolution_clock::now();
		// split node by whether cached or not 
		int64_t cached_cnt = 0;
		int64_t not_cached_cnt = 0;
		for(int i = 0; i < len; ++i){
			int64_t cur_node = node_id_data[i];
			int64_t req_idx = req_idx_data[i];
			
			// static cache lookup
			// fifo lookup
			int64_t gpu_idx = gpu_cache_node_map_data[cur_node];

			// dynamic cache lookup
			//int64_t gpu_idx = lfu_cache.access(cur_node);
			//int64_t gpu_idx = lru_cache.access(cur_node);
			if(gpu_idx != -1){
				cached_idx_on_cpu_data[cached_cnt] = gpu_idx;
				cached_req_idx_on_cpu_data[cached_cnt] = req_idx;
				cached_cnt ++;
			}
			else{
				not_cached_nid_on_cpu_data[not_cached_cnt] = cur_node;
				not_cached_req_idx_on_cpu_data[not_cached_cnt] = req_idx;
				not_cached_cnt ++;
			}
		}
		auto access_end = std::chrono::high_resolution_clock::now();
		auto access_us = std::chrono::duration_cast<std::chrono::microseconds>(access_end - access_start);
		//std::cout << "lru cache access takes " << access_us.count() << " us" << std::endl;
		//std::cout << "cached_cnt " << cached_cnt << " not_cached_cnt " << not_cached_cnt << std::endl;

		// resize temporary NDArray by actually size

		if(cached_cnt > 0){
			NDArray t_cached_idx_on_cpu = cached_idx_on_cpu.CreateView({cached_cnt}, {kDLInt, 64, 1});
			NDArray t_cached_idx_on_gpu = cached_idx_on_gpu.CreateView({cached_cnt}, {kDLInt, 64, 1});
			NDArray t_cached_req_idx_on_cpu = cached_req_idx_on_cpu.CreateView({cached_cnt}, {kDLInt, 64, 1});
			NDArray t_cached_req_idx_on_gpu = cached_req_idx_on_gpu.CreateView({cached_cnt}, {kDLInt, 64, 1});

			t_cached_idx_on_cpu.CopyTo(t_cached_idx_on_gpu, target_gpu_stream);
			t_cached_req_idx_on_cpu.CopyTo(t_cached_req_idx_on_gpu, target_gpu_stream);
			aten::IndexSelect2DAndScatter2D_(gpu_cache_buffer, t_cached_idx_on_gpu, ret, t_cached_req_idx_on_gpu);
		}

		if(not_cached_cnt == 0) {
			CUDA_CALL(cudaLaunchHostFunc(static_cast<cudaStream_t>(target_gpu_stream), check_caching_task_finished, static_cast<void*>(&sample_meta_data[worker_id][sample_idx%MAX_PENDING_SAMPLES])));
		}
	
		if(not_cached_cnt > 0){
			NDArray t_not_cached_nid_on_cpu = not_cached_nid_on_cpu.CreateView({not_cached_cnt}, {kDLInt, 64, 1});
			NDArray t_not_cached_new_idx_on_cpu = not_cached_new_idx_on_cpu.CreateView({not_cached_cnt}, {kDLInt, 64, 1});
			NDArray t_not_cached_new_idx_on_gpu = not_cached_new_idx_on_gpu.CreateView({not_cached_cnt}, {kDLInt, 64, 1});
			NDArray t_not_cached_req_idx_on_cpu = not_cached_req_idx_on_cpu.CreateView({not_cached_cnt}, {kDLInt, 64, 1});
			NDArray t_not_cached_req_idx_on_gpu = not_cached_req_idx_on_gpu.CreateView({not_cached_cnt}, {kDLInt, 64, 1});
			NDArray t_not_cached_feature_on_cpu = not_cached_feature_on_cpu.CreateView({not_cached_cnt, feature_dim}, {kDLFloat, 32, 1});
			NDArray t_not_cached_feature_on_gpu = not_cached_feature_on_gpu.CreateView({not_cached_cnt, feature_dim}, {kDLFloat, 32, 1});


			// for not cached nodes
			// fetch from CPU and remote machine
			// execute on CPU 
			aten::IndexSelect2DAndScatter2D_(cpu_cache_buffer, t_not_cached_nid_on_cpu, t_not_cached_feature_on_cpu);
			t_not_cached_req_idx_on_cpu.CopyTo(t_not_cached_req_idx_on_gpu, target_gpu_stream);
			t_not_cached_feature_on_cpu.CopyTo(t_not_cached_feature_on_gpu, target_gpu_stream);
			aten::Scatter2D_(t_not_cached_req_idx_on_gpu, t_not_cached_feature_on_gpu, ret); 
			CUDA_CALL(cudaLaunchHostFunc(static_cast<cudaStream_t>(target_gpu_stream), check_caching_task_finished, static_cast<void*>(&sample_meta_data[worker_id][sample_idx%MAX_PENDING_SAMPLES])));


			

			auto time_start = std::chrono::high_resolution_clock::now();
			// generate new new_idx for each node id using caching data structure
			for(int i = 0; i < not_cached_cnt; ++i){
				int64_t cur_node = not_cached_nid_on_cpu_data[i];
				/*direct mapping
				gpu_cache_node_map_data[cur_node] = cur_node / worker_num;
				not_cached_new_idx_on_cpu_data[i] = cur_node / worker_num;
				*/

				/*
				 * dynamic mapping
				 *
				if(gpu_cache_is_full){
					int64_t new_idx = lru_cache.evict_and_insert(cur_node);
					not_cached_new_idx_on_cpu_data[i] = new_idx;
					//lru_cache.insert(cur_node, new_idx);
				}
				else{
					not_cached_new_idx_on_cpu_data[i] = gpu_cache_cur_idx;
					lru_cache.insert(cur_node, gpu_cache_cur_idx);

					gpu_cache_cur_idx++;
					if(gpu_cache_cur_idx == cache_size_per_gpu)
						gpu_cache_is_full=true;
				}
				 *
				 */

				// fifo caching policy
				if(gpu_cache_is_full)
					gpu_cache_node_map_data[gpu_cache_reverse_idx_data[gpu_cache_cur_idx]] = -1;
				not_cached_new_idx_on_cpu_data[i] = gpu_cache_cur_idx;
				gpu_cache_node_map_data[cur_node] = gpu_cache_cur_idx;
				gpu_cache_reverse_idx_data[gpu_cache_cur_idx] = cur_node;
				gpu_cache_cur_idx++;
				if(gpu_cache_cur_idx == cache_size_per_gpu){
					gpu_cache_cur_idx = 0;
					gpu_cache_is_full = true;
				}

			}
			auto time_end = std::chrono::high_resolution_clock::now();
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
			//std::cout << "update lru cache using " << us.count() << " us" << std::endl;
			
			// according to new_idx, update gpu_cache_buffer 
			t_not_cached_new_idx_on_cpu.CopyTo(t_not_cached_new_idx_on_gpu, target_gpu_stream);
			aten::Scatter2D_(t_not_cached_new_idx_on_gpu, t_not_cached_feature_on_gpu, gpu_cache_buffer);

		}
		
	}


}

void start_caching_process(int worker_num_, int number_of_nodes_, int cache_size_per_gpu_, int feature_dim_, int max_inputs_length_, NDArray cpu_feature_tensor, int samples_per_worker_)
{
	// init some global 
	worker_num = worker_num_;
	number_of_nodes = number_of_nodes_;
	cache_size_per_gpu = cache_size_per_gpu_;
	feature_dim = feature_dim_;
	max_inputs_length = max_inputs_length_;
	samples_per_worker = samples_per_worker_;
	cpu_cache_buffer = cpu_feature_tensor;


	// this global gpu_cache_map is used by each working thread
	// lfu cache node map
	//gpu_cache_node_map = new dataNode * [number_of_nodes];
	//std::memset(gpu_cache_node_map, 0, sizeof(dataNode*) * number_of_nodes);
	//
	
	// lru cache node map
	// gpu_cache_node_map = new CacheNode * [number_of_nodes];
	// std::memset(gpu_cache_node_map, 0, sizeof(CacheNode*) * number_of_nodes);

	// static cache node map
	// fifo cache node map
	// cache node map for first four worker and GPU
	//std::cout <<"in start caching process" << std::endl;
	gpu_cache_node_map[0] = NDArray::Empty({number_of_nodes}, {kDLInt,64,1}, {kDLCPU, 0});
	std::memset(gpu_cache_node_map[0]->data, -1, gpu_cache_node_map[0].GetSize());

	// when workre_num is bigger than four,
	// we use another cache map for last four worker and GPU
	if(worker_num > 4){
		gpu_cache_node_map[1] = NDArray::Empty({number_of_nodes}, {kDLInt,64,1}, {kDLCPU, 0});
		std::memset(gpu_cache_node_map[1]->data, -1, gpu_cache_node_map[1].GetSize());
	}
	

	// open recv_fifo for each dispatching thread
	// open recv fifo with reading mode
	char recv_fifo_name[100];
	for(int i = 0; i < worker_num; ++i){
		sprintf(recv_fifo_name, "/tmp/recv_fifo_w%d",i);
		recv_fifo_fds[i] = open(recv_fifo_name, O_RDONLY);
	}

	// open cache_fifo for each working thread
	// open cache fifo with writing mode
	char cache_fifo_name[100];
	for(int i = 0; i < worker_num; ++i){
		sprintf(cache_fifo_name, "/tmp/cache_fifo_w%d",i);
		cache_fifo_fds[i] = open(cache_fifo_name, O_WRONLY);
	}

	//std::cout << "finish open fifio" << std::endl;

	std::vector<std::thread> dispatching_threads;
	std::vector<std::thread> working_threads;
	for(int i = 0; i < worker_num; ++i)
		dispatching_threads.push_back(std::thread(caching_dispatching_thread,i));
	for(int i = 0; i < worker_num; ++i)
		working_threads.push_back(std::thread(caching_working_thread, i));

	
	for(int i = 0; i < worker_num; ++i)
		dispatching_threads[i].join();
	for(int i = 0; i < worker_num; ++i)
		working_threads[i].join();
	

	for(int i = 0; i < worker_num; ++i)
		close(recv_fifo_fds[i]);
	for(int i = 0; i < worker_num; ++i)
		close(cache_fifo_fds[i]);
}

////////////////////////////////////////
////////////////////////////////////////
//

void * training_sample_feature_tensor_ptr[MAX_WORKER][MAX_PENDING_SAMPLES];
NDArray training_sample_feature_tensor[MAX_WORKER][MAX_PENDING_SAMPLES];
NDArray training_sample_label_tensor[MAX_WORKER][MAX_PENDING_SAMPLES];
SampleMetaData training_sample_meta_data[MAX_WORKER][MAX_PENDING_SAMPLES];
NDArray label_tensor;
NDArray nnz_buffer;



// recv thread enqueues sample results
// training process dequeue sample result and return to python layer
dmlc::moodycamel::BlockingConcurrentQueue<List<ObjectRef>> training_sample_result_queue;

void training_recv_thread(int worker_id)
{
	nnz_buffer = NDArray::Empty({max_inputs_length}, {kDLInt, 64, 1}, {kDLGPU, worker_id});

	//std::cout << "start training_recv_thread " << worker_id << std::endl;
	bool use_label = true;
	// label tensor is on CPU memory
	// thus needs to copy to GPU memory

	// define whether training process use label
	char cache_fifo_name[100];
	sprintf(cache_fifo_name, "/tmp/cache_fifo_w%d",worker_id);
	int cache_fifo_fd = open(cache_fifo_name, O_RDONLY);


	// managing graph structures memory on GPU
	bool preallocate = true;
	std::vector<NDArray> csr_indptr[MAX_PENDING_SAMPLES][MAX_LAYERS];
	std::vector<NDArray> csr_indices[MAX_PENDING_SAMPLES][MAX_LAYERS];
	std::vector<NDArray> csr_data[MAX_PENDING_SAMPLES][MAX_LAYERS];
	int real_layers = layers;
	
	// if label_tensor is null array, we should pre-allocate for anothre two graph: pos_graph and neg_graph
	// so, we have layers + 2 graphs
	if(dgl::aten::IsNullArray(label_tensor))
		real_layers = layers + 2;
	if (preallocate) {
		int csr_shape = max_inputs_length * 3;
		// init csr ndarray
		for(int i = 0; i < MAX_PENDING_SAMPLES; i++) {
			for(int j = 0; j < real_layers; j++) {
				// in_csr and out_csr
				csr_indptr[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));
				csr_indptr[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));

				csr_indices[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));
				csr_indices[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));

				csr_data[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));
				csr_data[i][j].push_back(NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id}));
				
				//csr_data[i][j] = NDArray::Empty({csr_shape}, {kDLInt, 64, 1}, {kDLGPU, worker_id});
			}
		}
		int total_shape = csr_shape * 3 * real_layers * MAX_PENDING_SAMPLES;
		std::cout << "Pre-allocate CudaMem for graph structures: " << csr_shape << " * " 
				<< 3 * real_layers * MAX_PENDING_SAMPLES * 2 << " = " << total_shape << ", size:" 
				<< (float) total_shape * 8 * 2 / 1024 / 1024<< " MB.\n";
	}

	cudaSetDevice(worker_id);
	//std::cout << "try to recv cuda mem ipc" << std::endl;
	// firstly, recv cudaIpcMemHandle from caching process and create 
	cudaIpcMemHandle_t cache_cuda_mem_ipc_handles[MAX_PENDING_SAMPLES];
	read_nbytes(cache_fifo_fd, static_cast<void*>(&cache_cuda_mem_ipc_handles), MAX_PENDING_SAMPLES*sizeof(cudaIpcMemHandle_t));
	for(int i = 0; i < MAX_PENDING_SAMPLES; ++i){
		cudaIpcOpenMemHandle(training_sample_feature_tensor_ptr[worker_id]+i, cache_cuda_mem_ipc_handles[i], cudaIpcMemLazyEnablePeerAccess);
		training_sample_feature_tensor[worker_id][i] = NDArray::EmptySharedGPU({max_inputs_length,feature_dim}, {kDLFloat,32, 1}, {kDLGPU, worker_id}, training_sample_feature_tensor_ptr[worker_id][i]);
		training_sample_label_tensor[worker_id][i] = NDArray::Empty({batch_size}, {kDLInt, 64, 1}, {kDLGPU, worker_id});
	}
	//std::cout << "training recv thread " << worker_id <<std::endl;

	for(int s = 0; s < samples_per_worker; ++s){
		int cur_sample_idx = s % MAX_PENDING_SAMPLES;
		SampleMetaData & smd = training_sample_meta_data[worker_id][cur_sample_idx];

		// DGL List object to store all object should returned to python layer
		List<ObjectRef> ret_list;


		
		auto access_start = std::chrono::high_resolution_clock::now();
		// receive sample meta data from corresponding cache fifo
		read_nbytes(cache_fifo_fd, static_cast<void*>(&smd), sizeof(SampleMetaData));
		auto access_end = std::chrono::high_resolution_clock::now();
		auto access_us = std::chrono::duration_cast<std::chrono::microseconds>(access_end - access_start);
		//std::cout << "------ training recv thread  " << worker_id << "takes " << access_us.count() << " us" << std::endl;
		/*
		auto tp = std::chrono::system_clock::now();
		std::time_t tt = std::chrono::system_clock::to_time_t(tp);
		std::cout << "training recv thread get info in " << tt << std::endl;
		*/


		//std::cout << "in trianing process " << smd.worker_id << " " << smd.sample_idx << std::endl;
		
		


		for(int l = 0; l < real_layers; ++l){
			char graph_name[100];
			sprintf(graph_name, "w%ds%d_b%d", worker_id, s, l);
			
			// get graph from shared memory
			HeteroGraphPtr hg;
			std::vector<std::string> ntypes;
			std::vector<std::string> etypes;
			std::tie(hg, ntypes, etypes) = HeteroGraph::CreateFromSharedMem(std::string(graph_name));

			// move graph to target gpu
			HeteroGraphPtr hg_new;
			if (!preallocate){
				hg_new = HeteroGraph::CopyTo(hg, DLContext{kDLGPU, worker_id});
			} else {
				hg_new = HeteroGraph::CopyTo(hg, csr_indptr[cur_sample_idx][l], 
										csr_indices[cur_sample_idx][l], csr_data[cur_sample_idx][l]);
			}
			// store all object in List object
			List<Value> ntypes_list, etypes_list;
			for(const auto & ntype : ntypes)
				ntypes_list.push_back(Value(MakeValue(ntype)));
			for(const auto & etype : etypes)
				etypes_list.push_back(Value(MakeValue(etype)));
			ret_list.push_back(HeteroGraphRef(hg_new));
			ret_list.push_back(ntypes_list);
			ret_list.push_back(etypes_list);
		}	

		int inputs_length = smd.inputs_length;
		NDArray training_sample_return_feature = training_sample_feature_tensor[worker_id][cur_sample_idx].CreateView({inputs_length, feature_dim}, {kDLFloat, 32, 1});
		ret_list.push_back(Value(MakeValue(training_sample_return_feature)));
		
		// if label_tensor is not null, we need to get label tensor
		if(!dgl::aten::IsNullArray(label_tensor)){
			char seeds_name[100];
			sprintf(seeds_name, "w%ds%d_seeds", worker_id, s);
			NDArray seed_nodes = NDArray::EmptyShared(std::string(seeds_name), {batch_size}, {kDLInt, 64, 1}, {kDLCPU, 0}, false);
			aten::IndexSelect(label_tensor, seed_nodes).CopyTo(training_sample_label_tensor[worker_id][cur_sample_idx]);
			NDArray temp_label_tensor = training_sample_label_tensor[worker_id][cur_sample_idx].CreateView({batch_size}, {kDLInt, 64, 1});
			ret_list.push_back(Value(MakeValue(temp_label_tensor)));
		}

		training_sample_result_queue.enqueue(ret_list);
	}
}


void start_training_recv_thread(int worker_id, int max_inputs_length_, int feature_dim_, int layers_, int batch_size_, NDArray labels, int samples_per_worker_)	
{
	max_inputs_length = max_inputs_length_;
	feature_dim = feature_dim_;
	layers = layers_;
	batch_size = batch_size_;
	label_tensor = labels;
	samples_per_worker = samples_per_worker_;

	std::thread training_recv_thread_ = std::thread(training_recv_thread, worker_id);
	training_recv_thread_.detach();
}

void set_dgl_kernel_stream(int target_gpu_id, int64_t torch_cuda_stream)
{
	DGLStreamHandle target_gpu_stream = reinterpret_cast<DGLStreamHandle>(torch_cuda_stream);
	std::cout << target_gpu_stream << std::endl;
	DGLSetStream(kDLGPU, target_gpu_id, target_gpu_stream);
	global_torch_cuda_stream = static_cast<cudaStream_t>(target_gpu_stream);
}


/// used in receiving process
///
///
//
//
//
//
//
//



struct SharedMemoryTask
{
	int worker_id;
	int sample_idx;
	List<HeteroGraphRef> hg_list;
	NDArray inputs;
	NDArray seeds;
	SharedMemoryTask(){}
	SharedMemoryTask(int wi, int si, List<HeteroGraphRef> hl, NDArray in, NDArray se): worker_id(wi), sample_idx(si), hg_list(hl), inputs(in), seeds(se){}
};

struct SharedMemorySampleResult
{
	int worker_id;
	int sample_idx;
	std::vector<HeteroGraphPtr> graph_vec;
	NDArray inputs;
	NDArray seeds;
	SharedMemorySampleResult(){}
	SharedMemorySampleResult(int wi, int si, std::vector<HeteroGraphPtr> gv, NDArray in, NDArray se):worker_id(wi), sample_idx(si), graph_vec(gv), inputs(in), seeds(se){}
};



#define MAX_MOVING_THREAD 5

dmlc::moodycamel::BlockingConcurrentQueue<SharedMemoryTask> shared_memory_task_queue[MAX_MOVING_THREAD]; 
dmlc::moodycamel::BlockingConcurrentQueue<SharedMemorySampleResult> shared_memory_finish_queue[MAX_MOVING_THREAD]; 
dmlc::moodycamel::BlockingConcurrentQueue<SharedMemorySampleResult> receiving_sample_result_queue; 
std::atomic<int> receiving_sample_result_queue_size;
std::atomic<int> receiving_remove_shared_memory_thread_finish_event;
std::vector<std::string> block_ntypes_vec, block_etypes_vec;
std::vector<std::string> graph_ntypes_vec, graph_etypes_vec;
std::set<std::string> fmts_set;
int num_moving_thread;

void receiving_move_to_shared_memory_thread(int thread_idx, int num_task)
{
	for(int i = 0; i < num_task; ++i){
		SharedMemoryTask smt;
		shared_memory_task_queue[thread_idx].wait_dequeue(smt);
		char name_buffer[100];
		sprintf(name_buffer, "w%ds%d_inputs", smt.worker_id, smt.sample_idx);
		NDArray new_inputs = NDArray::EmptyShared(std::string(name_buffer), {smt.inputs->shape[0]}, smt.inputs->dtype, smt.inputs->ctx, true);
		new_inputs.CopyFrom(smt.inputs);

		sprintf(name_buffer, "w%ds%d_seeds", smt.worker_id, smt.sample_idx);
		NDArray new_seeds = NDArray::EmptyShared(std::string(name_buffer), {smt.seeds->shape[0]}, smt.seeds->dtype, smt.seeds->ctx, true);
		new_seeds.CopyFrom(smt.seeds);

		std::vector<HeteroGraphPtr> graph_vec;
		for(int j = 0; j < layers; ++j){
			sprintf(name_buffer, "w%ds%d_b%d", smt.worker_id, smt.sample_idx, j);
			HeteroGraphPtr hg_shared = HeteroGraph::CopyToSharedMem(smt.hg_list[j].sptr(), std::string(name_buffer), block_ntypes_vec, block_etypes_vec, fmts_set);
			graph_vec.push_back(hg_shared);
		}
		
		// if the size of hg_list is bigger than number of layers, we are in unsupervised model and need to put pos_graph and neg_graph into shared memory
		// and append in blocks[layers], blocks[layers+1]
		if(smt.hg_list.size() > layers){
			sprintf(name_buffer, "w%ds%d_b%d", smt.worker_id, smt.sample_idx, layers);
			HeteroGraphPtr pos_shared = HeteroGraph::CopyToSharedMem(smt.hg_list[layers].sptr(), std::string(name_buffer), graph_ntypes_vec, graph_etypes_vec, fmts_set);
			graph_vec.push_back(pos_shared);

			sprintf(name_buffer, "w%ds%d_b%d", smt.worker_id, smt.sample_idx, layers+1);
			HeteroGraphPtr neg_shared = HeteroGraph::CopyToSharedMem(smt.hg_list[layers+1].sptr(), std::string(name_buffer), graph_ntypes_vec, graph_etypes_vec, fmts_set);
			graph_vec.push_back(neg_shared);
			
		}

		shared_memory_finish_queue[thread_idx].enqueue(SharedMemorySampleResult(smt.worker_id, smt.sample_idx, graph_vec, new_inputs, new_seeds));
	}
}

void receiving_send_signal_thread(int worker_id, int recv_fifo_fd)
{
	for(int i = 0; i < samples_per_worker; ++i){
		SharedMemorySampleResult smsr;
		shared_memory_finish_queue[i%num_moving_thread].wait_dequeue(smsr);
		SampleMetaData smd;
		smd.worker_id = worker_id;
		smd.sample_idx = i;
		smd.inputs_length = smsr.inputs->shape[0];
		smd.seeds_length = smsr.seeds->shape[0];

		// when sample_result_queue is full, this thread should wait
		while(receiving_sample_result_queue_size.load() == MAX_PENDING_SAMPLES){
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		write(recv_fifo_fd,  static_cast<void*>(&smd), sizeof(SampleMetaData));
		receiving_sample_result_queue.enqueue(SharedMemorySampleResult(smsr));
		receiving_sample_result_queue_size.fetch_add(1, std::memory_order_relaxed);
	}
}

void receiving_remove_shared_memory_thread(int worker_id, int train_fifo_fd)
{

	for(int i = 0; i < samples_per_worker; ++i){
		SampleMetaData smd;
		read_nbytes(train_fifo_fd, static_cast<void*>(&smd), sizeof(int) * 2);
		SharedMemorySampleResult smsr;
		receiving_sample_result_queue_size.fetch_sub(1, std::memory_order_relaxed);
		receiving_sample_result_queue.wait_dequeue(smsr);

		// implicit deletion of SharedMemorySampleResult
	}
	receiving_remove_shared_memory_thread_finish_event.fetch_add(1, std::memory_order_relaxed);

}

void start_receiving_children_thread(int worker_id, int layers_, int samples_per_worker_, int num_moving_thread_)
{
	layers = layers_;
	samples_per_worker = samples_per_worker_;
	num_moving_thread = num_moving_thread_;

	// static vector used in copy blocks into shared memory
	block_ntypes_vec.push_back(std::string("_U"));
	block_ntypes_vec.push_back(std::string("_U"));
	block_etypes_vec.push_back(std::string("_V"));

	// static vector used to copy regular graph
	graph_ntypes_vec.push_back(std::string("_U"));
	graph_etypes_vec.push_back(std::string("_V"));



	fmts_set.insert("csc");
	fmts_set.insert("csr");
	receiving_sample_result_queue_size = 0;
	receiving_remove_shared_memory_thread_finish_event = 0;

	char recv_fifo_name[100];
	sprintf(recv_fifo_name, "/tmp/recv_fifo_w%d",worker_id);
	int recv_fifo_fd = open(recv_fifo_name, O_WRONLY);

	char train_fifo_name[100];
	sprintf(train_fifo_name, "/tmp/train_fifo_w%d",worker_id);
	int train_fifo_fd = open(train_fifo_name, O_RDONLY);
	
	// start children threads of receiving process
	for(int i = 0; i < num_moving_thread; ++i){
		std::thread t = std::thread(receiving_move_to_shared_memory_thread, i, samples_per_worker / num_moving_thread);
		t.detach();
	}

	std::thread send_signal_thread = std::thread(receiving_send_signal_thread, worker_id, recv_fifo_fd);
	send_signal_thread.detach();

	std::thread remove_shared_memory_thread = std::thread(receiving_remove_shared_memory_thread, worker_id, train_fifo_fd);
	remove_shared_memory_thread.detach();


}
//////////////////////////////
//
// for DGL python call
//
//////////////////////////////
DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEStartCachingProcess")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
	int worker_num = args[0];
	int number_of_nodes = args[1];
	int cache_size_per_gpu = args[2];
	int feature_dim = args[3];
	int max_inputs_length = args[4];
	NDArray cpu_feature_tensor = args[5];
	int samples_per_worker = args[6];
	start_caching_process(worker_num,number_of_nodes,cache_size_per_gpu,feature_dim,max_inputs_length,cpu_feature_tensor,samples_per_worker);
});


DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEStartTrainingRecvThread")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
	int worker_id = args[0];
	int max_inputs_length= args[1];
	int feature_dim = args[2];
	int layers = args[3];
	int batch_size = args[4];
	NDArray label= args[5];
	int samples_per_worker = args[6];
	start_training_recv_thread(worker_id, max_inputs_length, feature_dim, layers, batch_size, label, samples_per_worker);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCESetTrainingDGLKernelStream")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
	int worker_id = args[0];
	int64_t torch_cuda_stream = args[1];
	set_dgl_kernel_stream(worker_id, torch_cuda_stream);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEGetSampleResult")
.set_body([] (DGLArgs args, DGLRetValue * rv){
	List<ObjectRef> ret;
	training_sample_result_queue.wait_dequeue(ret);
	*rv = ret;
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEStartReceivingChildrenThread")
.set_body([] (DGLArgs args, DGLRetValue * rv){
	int worker_id = args[0];
	int layers = args[1];
	int samples_per_worker = args[2];
	int num_moving_thread = args[3];
	start_receiving_children_thread(worker_id,layers,samples_per_worker,num_moving_thread);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEDispatchSharedMemoryTask")
.set_body([] (DGLArgs args, DGLRetValue * rv){
	int worker_id = args[0];
	int sample_idx = args[1];
	List<HeteroGraphRef> hg_list = args[2];
	NDArray inputs = args[3];
	NDArray seeds = args[4];
	shared_memory_task_queue[sample_idx % num_moving_thread].enqueue(SharedMemoryTask(worker_id, sample_idx, hg_list, inputs, seeds));
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEWaitReceivingRemoveSharedMemoryThread")
.set_body([] (DGLArgs args, DGLRetValue * rv){
	
	while(receiving_remove_shared_memory_thread_finish_event.load() == 0){
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
});


