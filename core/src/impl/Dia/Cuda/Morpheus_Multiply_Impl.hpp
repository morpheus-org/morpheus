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

#ifndef MORPHEUS_DIA_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Dia/Kernels/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    DiaTag, DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  using IndexType = typename LinearOperator::index_type;
  using ValueType = typename LinearOperator::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::spmv_dia_kernel<IndexType, ValueType, BLOCK_SIZE>, BLOCK_SIZE,
      (size_t)sizeof(IndexType) * BLOCK_SIZE);
  const size_t NUM_BLOCKS =
      std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.nrows(), BLOCK_SIZE));

  const IndexType* D     = A.diagonal_offsets.data();
  const ValueType* V     = A.values.data();
  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  const IndexType num_diagonals = A.values.ncols();
  const IndexType pitch         = A.values.nrows();

  if (num_diagonals == 0) {
    // empty matrix
    return;
  }

  Kernels::spmv_dia_kernel<IndexType, ValueType, BLOCK_SIZE>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.ncols(), num_diagonals,
                                      pitch, D, V, x_ptr, y_ptr);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DIA_CUDA_MULTIPLY_IMPL_HPP