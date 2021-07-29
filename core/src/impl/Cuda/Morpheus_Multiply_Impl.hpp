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

#ifndef MORPHEUS_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Cuda/Morpheus_Multiply_Kernels.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CooTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using IndexType = typename LinearOperator::index_type;
  using ValueType = typename LinearOperator::value_type;

  const IndexType* I = A.row_indices.data();
  const IndexType* J = A.column_indices.data());
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  Morpheus::Impl::Kernels::spmv_coo_serial_kernel<IndexType, ValueType>
      <<<1, 1>>>(A.nnnz(), I, J, V, x_ptr, y_ptr);
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CsrTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using IndexType = typename LinearOperator::index_type;
  using ValueType = typename LinearOperator::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (A.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const IndexType* I = A.row_offsets.data();
  const IndexType* J = A.column_indices.data());
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  Morpheus::Impl::Kernels::spmv_csr_scalar_kernel<IndexType, ValueType>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), I, J, V, x_ptr, y_ptr);
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    DiaTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using IndexType = typename LinearOperator::index_type;
  using ValueType = typename LinearOperator::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (A.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const IndexType* D     = A.diagonal_offsets.data());
  const ValueType* V     = A.values.data();
  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_OPENMP_MULTIPLY_IMPL_HPP