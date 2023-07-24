/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_index_select.cu
 * \brief Array index select GPU implementation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void _IndexSelect2DKernel(const DType* array, const IdType* index,
                                   int64_t length, int64_t dim,  DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    //out[tx] = array[index[tx]];
    int64_t dstOffset = tx * dim;
    int64_t srcOffset = index[tx] * dim;
    for(int64_t i = 0; i < dim; dstOffset++, srcOffset++, i++)
      out[dstOffset] = array[srcOffset]; 
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void _IndexSelect2DKernel_v2(const DType* array, const IdType* index,
                                        int64_t length, int64_t dim,  DType* out) {
  for(int tx = blockIdx.x * blockDim.x + threadIdx.x;
      tx < length; tx += gridDim.x * blockDim.x) {
    int dst_index = tx / dim;
    int ele_inslice = tx % dim; 
    int64_t srcOffset = index[dst_index] * dim + ele_inslice;
    int64_t dstOffset = tx;
    out[dstOffset] = array[srcOffset];
  }
}



template<DLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect2D(NDArray array, IdArray index) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t dim = array->shape[1];
  const int64_t len = index->shape[0] * dim;
  NDArray ret = NDArray::Empty({len,dim}, array->dtype, array->ctx);
  if (len == 0)
    return ret;
  DType* ret_data = static_cast<DType*>(ret->data);
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  //_IndexSelect2DKernel<<<nb, nt, 0, thr_entry->stream>>>(array_data, idx_data, len, dim, ret_data);
  _IndexSelect2DKernel_v2<<<nb, nt, 0, thr_entry->stream>>>(array_data, idx_data, len, dim, ret_data);
  return ret;
}


template NDArray IndexSelect2D<kDLGPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLGPU, double, int64_t>(NDArray, IdArray);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
