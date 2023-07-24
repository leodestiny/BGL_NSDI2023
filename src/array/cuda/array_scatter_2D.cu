/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cuda/array_scatter.cu
 * \brief Array scatter GPU implementation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void _Scatter2DKernel(const IdType* index, const DType* value,
                               int64_t length, int64_t dim, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int64_t dstOffset = index[tx] * dim;
    int64_t srcOffset = tx * dim;
    for(int64_t i = 0; i < dim; dstOffset++, srcOffset++, i++){
      //out[index[tx]] = value[tx];
      out[dstOffset] = value[srcOffset];
    }
    tx += stride_x;
  }
}


template <typename DType, typename IdType>
__global__ void _Scatter2DKernel_v2(const IdType* index, const DType* value,
                                    int64_t length, int64_t dim, DType* out) {
  for(int tx = blockIdx.x * blockDim.x + threadIdx.x;
    tx < length; tx += gridDim.x * blockDim.x) {
    int dst_index = tx / dim;
    int ele_inslice = tx % dim; 
    int64_t dstOffset = index[dst_index] * dim + ele_inslice;
    int64_t srcOffset = tx;
    out[dstOffset] = value[srcOffset];
  }
}

template <DLDeviceType XPU, typename DType, typename IdType>
void Scatter2D_(IdArray index, NDArray value, NDArray out) {
  const int64_t dim = value->shape[1];
  const int64_t len = index->shape[0] * dim;
  const IdType* idx = index.Ptr<IdType>();
  const DType* val = value.Ptr<DType>();
  DType* outd = out.Ptr<DType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  //_Scatter2DKernel<<<nb, nt, 0, thr_entry->stream>>>(idx, val, len, dim,outd);
  _Scatter2DKernel_v2<<<nb, nt, 0, thr_entry->stream>>>(idx, val, len, dim,outd);
}



template void Scatter2D_<kDLGPU, int32_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, int64_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, float, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, double, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, int32_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, int64_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, float, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLGPU, double, int64_t>(IdArray, NDArray, NDArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
