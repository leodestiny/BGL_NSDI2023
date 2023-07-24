/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_scatter.cc
 * \brief Array scatter CPU implementation
 */
#include <dgl/array.h>
#include <stdlib.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType, typename IdType>
void Scatter2D_(IdArray index, NDArray value, NDArray out) {
  const int64_t len = index->shape[0];
  const int64_t dim = value->shape[1];
  const int64_t row_size = dim * sizeof(DType);
  const IdType* idx = index.Ptr<IdType>();
  const DType* val = value.Ptr<DType>();
  DType* outd = out.Ptr<DType>();
#pragma omp parallel for
  for (int64_t i = 0; i < len; ++i){
   // outd[idx[i]] = val[i];
   memcpy(outd + idx[i] * dim , val + i * dim, row_size);

  }
}

template void Scatter2D_<kDLCPU, int32_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, int64_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, float, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, double, int32_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, int32_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, int64_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, float, int64_t>(IdArray, NDArray, NDArray);
template void Scatter2D_<kDLCPU, double, int64_t>(IdArray, NDArray, NDArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
