from .. import backend
from .._ffi.function import _init_api
import pickle
from .kvstore import KVSTORE_PULL
from ..heterograph import DGLBlock, DGLHeteroGraph
import time

def init_tensor_cache_engine(number_of_nodes, partition_num, gpu_cache_size, feature_dim, feature_name, kv_client):
    pickle_data = bytearray(pickle.dumps(([0], [feature_name])))

    _CAPI_DGLTCEInit(number_of_nodes,partition_num, gpu_cache_size, feature_dim,
            int(kv_client._machine_count), int(kv_client._group_count), int(kv_client._client_id), KVSTORE_PULL,
            pickle_data)


    def insert_partition_feature_tensor(part_id, feature_tensor):
        _CAPI_DGLTCEInsertPartitionFeatureTensor(part_id, backend.zerocopy_to_dgl_ndarray(feature_tensor))

def remove_partition_feature_tensor(part_id):
    _CAPI_DGLTCERemotePartitionFeatureTensor(part_id)


def record_global2partition(g2p):
    _CAPI_DGLTCERecordGlobal2Partition(backend.zerocopy_to_dgl_ndarray(g2p))


def record_global2local(g2l):
    _CAPI_DGLTCERecordGlobal2Local(backend.zerocopy_to_dgl_ndarray(g2l))


def gather_tensor_on_gpu(req_node_ids):
    return backend.zerocopy_from_dgl_ndarray(_CAPI_DGLTCEGatherTensorOnGPU(backend.zerocopy_to_dgl_ndarray(req_node_ids)))

def start_caching_process(worker_num, number_of_nodes, cache_size_per_gpu, feature_dim, max_inputs_length, cpu_feature, samples_per_worker):
    _CAPI_DGLTCEStartCachingProcess(worker_num, number_of_nodes, cache_size_per_gpu, feature_dim, max_inputs_length, backend.zerocopy_to_dgl_ndarray(cpu_feature), samples_per_worker)

def start_training_recv_thread(worker_id, max_inputs_length, feature_dim, layers, batch_size, label, samples_per_worker):
    _CAPI_DGLTCEStartTrainingRecvThread(worker_id, max_inputs_length, feature_dim, layers, batch_size, backend.zerocopy_to_dgl_ndarray(label), samples_per_worker)

def start_receiving_children_thread(worker_id, layers, samples_per_worker, num_moving_thread):
    _CAPI_DGLTCEStartReceivingChildrenThread(worker_id, layers, samples_per_worker, num_moving_thread)

def get_sample_result(layer_num, use_label=True):
    ret_list =  _CAPI_DGLTCEGetSampleResult()
    blocks = []
    # reconstruct blocks
    for i in range(layer_num):
        g, ntypes, etypes = ret_list[3*i:3*(i+1)]
        blocks.append(DGLBlock(g, ntypes, etypes))

    if use_label:
        feature = backend.zerocopy_from_dgl_ndarray(ret_list[3*layer_num])
        label = backend.zerocopy_from_dgl_ndarray(ret_list[3*layer_num+1])
        return blocks, feature, label
    else:
        pos_hg, pos_ntypes, pos_etypes = ret_list[3*(layer_num):3*(layer_num+1)]
        pos_graph = DGLHeteroGraph(pos_hg, pos_ntypes, pos_etypes)
        neg_hg, neg_ntypes, neg_etypes = ret_list[3*(layer_num+1):3*(layer_num+2)]
        neg_graph = DGLHeteroGraph(neg_hg, neg_ntypes, neg_etypes)
        feature = backend.zerocopy_from_dgl_ndarray(ret_list[3*(layer_num+2)])
        return blocks, pos_graph, neg_graph, feature

def dispatch_shared_memory_task(worker_id, sample_idx, blocks, input_nodes, seeds):
    hg_list = [b._graph for b in blocks]
    _CAPI_DGLTCEDispatchSharedMemoryTask(worker_id, sample_idx, hg_list, backend.zerocopy_to_dgl_ndarray(input_nodes), backend.zerocopy_to_dgl_ndarray(seeds))

def dispatch_unsuper_shared_memory_task(worker_id, sample_idx, blocks, pos_graph, neg_graph, input_nodes, seeds):
    hg_list = [b._graph for b in blocks]
    hg_list.append(pos_graph._graph)
    hg_list.append(neg_graph._graph)
    _CAPI_DGLTCEDispatchSharedMemoryTask(worker_id, sample_idx, hg_list, backend.zerocopy_to_dgl_ndarray(input_nodes), backend.zerocopy_to_dgl_ndarray(seeds))

def wait_receiving_remove_shared_memory_thread():
    _CAPI_DGLTCEWaitReceivingRemoveSharedMemoryThread()

def set_training_dgl_kernel_stream(worker_id, cuda_stream):
    _CAPI_DGLTCESetTrainingDGLKernelStream(worker_id, cuda_stream)

_init_api("dgl.distributed.tensor_cache_engine")
