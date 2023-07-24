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



using namespace dgl::runtime;
using namespace dgl::rpc;
using namespace dgl;


struct TensorCacheEngine{
  // variables used in rpc
  int machine_count;
  int group_count;
  int client_id;
  int service_id;
  std::string pickle_data;


  // variables used in cache engine
  int number_of_nodes;
  int gpu_cache_size;
  int tensor_dim;
  int gpu_cache_cur_idx;
  bool gpu_cache_is_full;
  int partition_num;


  // tensor buffer record real tensor
  NDArray gpu_cache_buffer;

  // map column index to global node id
  NDArray gpu_cache_reverse_idx;

  // record wether node has been cached on GPU
  // if cached, map from global node id to columen index
  // -1 means this node is not cached 
  NDArray gpu_cache_node_map;

  // record the partition_id of each node
  // size: {the number of nodes in entire graph}
  NDArray node_global2partition;


  // record the local index of each node in its partition
  // size: {the number of nodes in entire graph}
  // used in gather tensor on CPU
  NDArray node_global2local;

  std::vector<NDArray> cpu_cache_buffer;
  std::vector<bool> cpu_cached_partition;
  

  TensorCacheEngine(){}

  void Init(int number_of_n, int pn, int gpu_size, int dim, int mc, int gc, int ci, int si, std::string pd){
    machine_count = mc;
    group_count = gc;
    client_id = ci;
    service_id = si;
    pickle_data = pd;
    number_of_nodes = number_of_n;
    gpu_cache_size = gpu_size;
    tensor_dim = dim;
    partition_num = pn;

    cpu_cached_partition.resize(partition_num);
    for(int i = 0; i < partition_num; ++i)
      cpu_cached_partition[i] = false;
    cpu_cache_buffer.resize(partition_num);


    gpu_cache_cur_idx = 0;
    gpu_cache_is_full = false;
    gpu_cache_buffer = NDArray::Empty({gpu_size,dim}, {kDLFloat,32,1}, {kDLGPU,0});
    gpu_cache_reverse_idx = NDArray::Empty({gpu_size}, {kDLInt,64,1}, {kDLCPU,0});
    gpu_cache_node_map = NDArray::Empty({number_of_nodes}, {kDLInt,64,1}, {kDLCPU,0});
    
    // memset all node_map with -1, means no cache 
    std::memset(gpu_cache_node_map->data, -1, gpu_cache_node_map.GetSize());
    
  }
  
  void RecordGlobal2Partition(NDArray global2part){
    node_global2partition = global2part;
  }

  void RecordGlobal2Local(NDArray global2local){
    node_global2local = global2local;
  }
  
  void InsertPartitionFeatureTensor(int part_id, NDArray partition_feature_tensor){
    cpu_cached_partition[part_id] = true;
    cpu_cache_buffer[part_id] = partition_feature_tensor;
  }
  
  void RemovePartitionFeatureTensor(int part_id){
    cpu_cached_partition[part_id] = false;
    NDArray empty;
    cpu_cache_buffer[part_id] = empty;
  }


  NDArray GatherTensorOnGPU(IdArray req_node_ids){
    const int64_t len = req_node_ids->shape[0];

    // [0 to partition_num - 1] for count real value in each partition
    int part_count[partition_num];
    int gpu_count = 0;
    for(int i = 0; i < partition_num; i++)
      part_count[i] = 0;
       
    
    // we allocate a vector with fixed length to reduce the allocation cost
    // and adjust by count when finish
    std::vector<int64_t> gpu_cached_idx(len);
    std::vector<int64_t> gpu_cached_req_idx(len);
    std::vector<std::vector<int64_t> > node_ids_by_part(partition_num, std::vector<int64_t>(len));
    std::vector<std::vector<int64_t> > node_req_idx_by_part(partition_num,std::vector<int64_t>(len));

    NDArray gpu_cached_tensor, cpu_cached_tensor, remote_tensor;
    NDArray ret = NDArray::Empty({len,tensor_dim}, {kDLFloat,32,1}, {kDLGPU,0});


    const int64_t * req_node_ids_data = static_cast<int64_t*>(req_node_ids->data);
    int64_t * gpu_cache_node_map_data = static_cast<int64_t*>(gpu_cache_node_map->data);
    int64_t * gpu_cache_reverse_idx_data = static_cast<int64_t*>(gpu_cache_reverse_idx->data);
    int64_t * node_global2partition_data = static_cast<int64_t*>(node_global2partition->data);
    int64_t * node_global2local_data = static_cast<int64_t*>(node_global2local->data);

    for(int64_t i = 0; i < len; ++i){
      int64_t cur_node = req_node_ids_data[i];
      int64_t gpu_idx = gpu_cache_node_map_data[cur_node];

      if(gpu_idx != -1){
        gpu_cached_idx[gpu_count] = gpu_idx;
	gpu_cached_req_idx[gpu_count] = i;
	gpu_count++;
      }
      else{
	int p_id = node_global2partition_data[cur_node];
	// some node with partition id -1, should not processed
	if(p_id == -1)
	  continue;
	int pc = part_count[p_id]++;
	node_ids_by_part[p_id][pc] = cur_node;
	node_req_idx_by_part[p_id][pc] = i;
      }
    }

    std::cout << "gpu cache ratio " << 100.0 * gpu_count / len << "%" << std::endl;
    // resize vector by real size
    gpu_cached_idx.resize(gpu_count);
    gpu_cached_req_idx.resize(gpu_count);
    for(int i = 0; i < partition_num; ++i){
      int pc = part_count[i];
      node_ids_by_part[i].resize(pc);
      node_req_idx_by_part[i].resize(pc);
      std::cout << "part id" << i << " ratio "  << pc * 100.0 / len << " %" << std::endl;
    }


      // Send node ids which are not cached in cpu buffer to remote graph store 
      int msg_count = 0;
      int msg_seq = (RPCContext::ThreadLocal()->msg_seq)++;
      for (int i = 0; i < partition_num; ++i) {
        if (node_ids_by_part[i].size() != 0 && ! cpu_cached_partition[i]) {
          RPCMessage msg;
          msg.service_id = service_id;
          msg.msg_seq = msg_seq;
          msg.client_id = client_id;
          int lower = i*group_count;
          int upper = (i+1)*group_count;
          msg.server_id = dgl::RandomEngine::ThreadLocal()->RandInt(lower, upper);
          msg.data = pickle_data;
          NDArray tensor = NDArray::FromVector(node_ids_by_part[i]);
          msg.tensors.push_back(tensor);
          SendRPCMessage(msg, msg.server_id);
          msg_count++;
        }
      }


    if(gpu_cached_idx.size() > 0){
      gpu_cached_tensor = aten::IndexSelect2D(gpu_cache_buffer, NDArray::FromVector(gpu_cached_idx).CopyTo(gpu_cache_buffer->ctx));
      if(gpu_cached_idx.size() == len)
        return gpu_cached_tensor;
      aten::Scatter2D_(NDArray::FromVector(gpu_cached_req_idx).CopyTo(gpu_cache_buffer->ctx), gpu_cached_tensor, ret);
    }


    for(int i = 0; i < partition_num; ++i){
      if(node_ids_by_part[i].size() != 0 && cpu_cached_partition[i]){
	 auto v = node_ids_by_part[i];
	 int part_len = v.size();
	 std::vector<int64_t> cpu_cached_idx(part_len);
	 std::vector<int64_t> new_gpu_cache_idx(part_len);
	 for(int j = 0; j < part_len; ++j){
	   auto cur_node = v[j];
	   cpu_cached_idx[j] = node_global2local_data[cur_node];
           new_gpu_cache_idx[j] = gpu_cache_cur_idx;
	   if(gpu_cache_is_full)
	     gpu_cache_node_map_data[gpu_cache_reverse_idx_data[gpu_cache_cur_idx]] = -1;
	   gpu_cache_node_map_data[cur_node] = gpu_cache_cur_idx;
	  
	   gpu_cache_cur_idx++;
	   if(gpu_cache_cur_idx == gpu_cache_size){
	     gpu_cache_cur_idx = 0;
	     gpu_cache_is_full = true;
	   }
	 }

	 // gather tensor from cpu cache buffer and write to final return tensor
	 NDArray cpu_cached_tensor = aten::IndexSelect2D(cpu_cache_buffer[i], NDArray::FromVector(cpu_cached_idx)).CopyTo(gpu_cache_buffer->ctx);
         aten::Scatter2D_(NDArray::FromVector(node_req_idx_by_part[i]).CopyTo(gpu_cache_buffer->ctx), cpu_cached_tensor, ret);
	 
	 // update gpu_cache_buffer
         aten::Scatter2D_(NDArray::FromVector(new_gpu_cache_idx).CopyTo(gpu_cache_buffer->ctx), cpu_cached_tensor, gpu_cache_buffer);
      }
    }
    
    
    if(msg_count > 0){
      for (int i = 0; i < msg_count; ++i) {
        RPCMessage msg;
        RecvRPCMessage(&msg, 0);
        int part_id = msg.server_id / group_count;
	NDArray remote_tensor = msg.tensors[0].CopyTo(gpu_cache_buffer->ctx);

        aten::Scatter2D_(NDArray::FromVector(node_req_idx_by_part[part_id]).CopyTo(gpu_cache_buffer->ctx), remote_tensor , ret);

	auto v = node_ids_by_part[part_id]; 
	int part_len = v.size();
        std::vector<int64_t> new_gpu_cache_idx(part_len);
        for(int64_t j = 0; j < part_len; j ++){
          int64_t cur_node = v[j];
          new_gpu_cache_idx[j] = gpu_cache_cur_idx;
          if(gpu_cache_is_full)
            gpu_cache_node_map_data[gpu_cache_reverse_idx_data[gpu_cache_cur_idx]] = -1;
          gpu_cache_node_map_data[cur_node] = gpu_cache_cur_idx;
            
          gpu_cache_cur_idx++;
          if(gpu_cache_cur_idx == gpu_cache_size){
            gpu_cache_cur_idx = 0;
            gpu_cache_is_full = true;
          }
        }
        aten::Scatter2D_(NDArray::FromVector(new_gpu_cache_idx).CopyTo(gpu_cache_buffer->ctx), remote_tensor, gpu_cache_buffer);
      }
    }
    return ret;
  }
  static TensorCacheEngine* ThreadLocal(){
    return dmlc::ThreadLocalStore<TensorCacheEngine>::Get();
  }




  
};



DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEInit")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int number_of_nodes = args[0];
  int partition_num = args[1];
  int gpu_size = args[2];
  int dim = args[3];
  int machine_count = args[4];
  int group_count = args[5];
  int client_id = args[6];
  int service_id = args[7];
  std::string pickle_data = args[8];
  TensorCacheEngine::ThreadLocal()->Init(number_of_nodes, partition_num, gpu_size, dim, machine_count, group_count, client_id, service_id, pickle_data);
});



DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEInsertPartitionFeatureTensor")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int part_id = args[0];
  NDArray partition_feature_tensor = args[1];
  TensorCacheEngine::ThreadLocal()->InsertPartitionFeatureTensor(part_id, partition_feature_tensor);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCERemovePartitionFeatureTensor")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int part_id = args[0];
  TensorCacheEngine::ThreadLocal()->RemovePartitionFeatureTensor(part_id);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCERecordGlobal2Partition")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray global2partition = args[0];
  TensorCacheEngine::ThreadLocal()->RecordGlobal2Partition(global2partition);
});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCERecordGlobal2Local")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray global2local = args[0];
  TensorCacheEngine::ThreadLocal()->RecordGlobal2Local(global2local);

});

DGL_REGISTER_GLOBAL("distributed.tensor_cache_engine._CAPI_DGLTCEGatherTensorOnGPU")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  IdArray req_node_ids = args[0];
  *rv = TensorCacheEngine::ThreadLocal()->GatherTensorOnGPU(req_node_ids);
});
