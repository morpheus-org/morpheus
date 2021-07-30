/**
 * Morpheus_Multiply_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
 *
 * Contributing Authors:
 * Christodoulos Stylianou (c.stylianou@ed.ac.uk)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

namespace Kernels {

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a
// larger matrix
template <typename IndexType, typename ValueType>
MORPHEUS_INLINE_FUNCTION void spmv_coo_serial_kernel(
    const IndexType nnnz, const IndexType* I, const IndexType* J,
    const ValueType* V, const ValueType* x, ValueType* y) {
  for (IndexType n = 0; n < nnnz; n++) {
    y[I[n]] += V[n] * x[J[n]];
  }
}

}  // namespace Kernels

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CooTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using IndexType    = typename LinearOperator::index_type;
  using ValueType    = typename LinearOperator::value_type;
  const IndexType* I = A.row_indices.data();
  const IndexType* J = A.column_indices.data();
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  Morpheus::Impl::Kernels::spmv_coo_serial_kernel<IndexType, ValueType>
      <<<1, 1>>>(A.nnnz(), I, J, V, x_ptr, y_ptr);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP