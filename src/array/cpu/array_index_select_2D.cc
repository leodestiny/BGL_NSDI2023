/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_index_select.cc
 * \brief Array index select CPU implementation
 */
#include <dgl/array.h>
#include <stdlib.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template<DLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect2D(NDArray array, IdArray index) {
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  const int64_t dim = array->shape[1];
  const int64_t row_size = dim * sizeof(DType);
  NDArray ret = NDArray::Empty({len,dim}, array->dtype, array->ctx);
  DType* ret_data = static_cast<DType*>(ret->data);
 #pragma omp parallel for
  for (int64_t i = 0; i < len; ++i) {
    CHECK_LT(idx_data[i], arr_len) << "Index out of range.";
    //ret_data[i] = array_data[idx_data[i]];
    memcpy(ret_data + i * dim, array_data + idx_data[i] * dim, row_size);
  }
  return ret;
}

template NDArray IndexSelect2D<kDLCPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect2D<kDLCPU, double, int64_t>(NDArray, IdArray);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
