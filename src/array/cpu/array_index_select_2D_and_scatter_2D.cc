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
void IndexSelect2DAndScatter2D_(NDArray array, IdArray index, NDArray out) {
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  const int64_t dim = array->shape[1];
  const int64_t row_size = dim * sizeof(DType);
  DType* out_data = static_cast<DType*>(out->data);
 #pragma omp parallel for
  for (int64_t i = 0; i < len; ++i) {
    CHECK_LT(idx_data[i], arr_len) << "Index out of range.";
    //ret_data[i] = array_data[idx_data[i]];
    memcpy(out_data + i * dim, array_data + idx_data[i] * dim, row_size);
  }
}

template void IndexSelect2DAndScatter2D_<kDLCPU, int32_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int32_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int64_t, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int64_t, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, float, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, float, int64_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, double, int32_t>(NDArray, IdArray, NDArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, double, int64_t>(NDArray, IdArray,NDArray);

template<DLDeviceType XPU, typename DType, typename IdType>
void IndexSelect2DAndScatter2D_(NDArray array, IdArray index, NDArray out, IdArray out_index) {
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const IdType* out_idx_data = static_cast<IdType*>(out_index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  const int64_t dim = array->shape[1];
  const int64_t row_size = dim * sizeof(DType);
  DType* out_data = static_cast<DType*>(out->data);
 #pragma omp parallel for
  for (int64_t i = 0; i < len; ++i) {
    CHECK_LT(idx_data[i], arr_len) << "Index out of range.";
    //ret_data[i] = array_data[idx_data[i]];
    memcpy(out_data + out_idx_data[i] * dim, array_data + idx_data[i] * dim, row_size);
  }
}

template void IndexSelect2DAndScatter2D_<kDLCPU, int32_t, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int32_t, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int64_t, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, int64_t, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, float, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, float, int64_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, double, int32_t>(NDArray, IdArray, NDArray, IdArray);
template void IndexSelect2DAndScatter2D_<kDLCPU, double, int64_t>(NDArray, IdArray,NDArray, IdArray);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
