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
__global__ void _IndexSelect2DAndScatter2DKernel(const DType* in, const IdType* in_idx,
                                   int64_t length, int64_t dim,  DType* out, IdType* out_idx) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    //out[tx] = array[index[tx]];
    int64_t srcOffset = in_idx[tx] * dim;
    int64_t dstOffset = out_idx[tx] * dim;
    for(int64_t i = 0; i < dim; dstOffset++, srcOffset++, i++)
      out[dstOffset] = in[srcOffset]; 
    tx += stride_x;
  }
}


template <typename DType, typename IdType>
__global__ void _IndexSelect2DAndScatter2DKernel_v2(const DType* in, const IdType* in_idx,
                                                    int64_t length, int64_t dim,  DType* out, IdType* out_idx) {
  for(int tx = blockIdx.x * blockDim.x + threadIdx.x;
      tx < length; tx += gridDim.x * blockDim.x) {
    int dst_index = tx / dim;
    int ele_inslice = tx % dim; 
    int64_t srcOffset = in_idx[dst_index] * dim + ele_inslice;
    int64_t dstOffset = out_idx[dst_index] * dim + ele_inslice;
    out[dstOffset] = in[srcOffset];
  }
}


template<DLDeviceType XPU, typename DType, typename IdType>
void IndexSelect2DAndScatter2D_(NDArray in, IdArray in_index, NDArray out, IdArray out_index) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* in_data = static_cast<DType*>(in->data);
  const IdType* in_idx_data = static_cast<IdType*>(in_index->data);
  DType* out_data = static_cast<DType*>(out->data);
  IdType* out_idx_data = static_cast<IdType*>(out_index->data);
  const int64_t dim = in->shape[1];
  const int64_t len = in_index->shape[0] * dim;
  const int nt = cuda::FindNumThreads(len);

  const int nb = (len + nt - 1) / nt;
  _IndexSelect2DAndScatter2DKernel_v2<<<nb, nt, 0, thr_entry->stream>>>(in_data, in_idx_data, len, dim, out_data, out_idx_data);
  //_IndexSelect2DAndScatter2DKernel<<<nb, nt, 0, thr_entry->stream>>>(in_data, in_idx_data, len, dim, out_data, out_idx_data);
}


template void IndexSelect2DAndScatter2D_<kDLGPU, int32_t, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int32_t, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int64_t, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int64_t, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, float, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, float, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, double, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, double, int64_t>(NDArray, IdArray, NDArray, IdArray);

template <typename DType, typename IdType>
__global__ void _IndexSelect2DAndScatter2DKernel(const DType* in, const IdType* in_idx,
                                   int64_t length, int64_t dim,  DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    //out[tx] = array[index[tx]];
    int64_t srcOffset = in_idx[tx] * dim;
    int64_t dstOffset = tx * dim;
    for(int64_t i = 0; i < dim; dstOffset++, srcOffset++, i++)
      out[dstOffset] = in[srcOffset]; 
    tx += stride_x;
  }
}

template<DLDeviceType XPU, typename DType, typename IdType>
void IndexSelect2DAndScatter2D_(NDArray in, IdArray in_index, NDArray out) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* in_data = static_cast<DType*>(in->data);
  const IdType* in_idx_data = static_cast<IdType*>(in_index->data);
  DType* out_data = static_cast<DType*>(out->data);
  const int64_t len = in_index->shape[0];
  const int64_t dim = in->shape[1];
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  _IndexSelect2DAndScatter2DKernel<<<nb, nt, 0, thr_entry->stream>>>(in_data, in_idx_data, len, dim, out_data);
}

template void IndexSelect2DAndScatter2D_<kDLGPU, int32_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int32_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int64_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, int64_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, float, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, float, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, double, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLGPU, double, int64_t>(NDArray, IdArray, NDArray);



}  // namespace impl
}  // namespace aten
}  // namespace dgl
