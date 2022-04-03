/**
 * Morpheus_MatrixOperations_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_DIA_CUDA_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DIA_CUDA_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Dia/Kernels/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, DiaTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using index_type = typename SparseMatrix::index_type;
  using value_type = typename SparseMatrix::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::update_dia_diagonal_kernel<value_type, index_type>, BLOCK_SIZE,
      (size_t)sizeof(index_type) * BLOCK_SIZE);
  const size_t NUM_BLOCKS =
      std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.nrows(), BLOCK_SIZE));

  const index_type num_diagonals = A.cvalues().ncols();
  const index_type pitch         = A.cvalues().nrows();

  Kernels::update_dia_diagonal_kernel<value_type, index_type>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.ncols(), num_diagonals,
                                      pitch, A.diagonal_offsets().data(),
                                      A.values().data(), diagonal.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("update_dia_diagonal_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DIA_CUDA_MATRIXOPERATIONS_IMPL_HPP
