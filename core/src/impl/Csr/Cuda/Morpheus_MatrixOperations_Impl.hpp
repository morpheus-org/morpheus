/**
 * Morpheus_MatrixOperations_Impl.hpp
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

#ifndef MORPHEUS_CSR_CUDA_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_CSR_CUDA_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Csr/Kernels/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, CsrTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using IndexType = typename SparseMatrix::index_type;
  using ValueType = typename SparseMatrix::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (A.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Kernels::update_csr_diagonal_kernel<ValueType, IndexType>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.row_offsets().data(),
                                      A.column_indices().data(),
                                      A.values().data(), diagonal.data());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_CSR_CUDA_MATRIXOPERATIONS_IMPL_HPP
