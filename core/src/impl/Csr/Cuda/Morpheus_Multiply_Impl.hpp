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

#ifndef MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Csr/Kernels/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

// forward decl
template <size_t THREADS_PER_VECTOR, typename Matrix, typename Vector>
void __spmv_csr_vector(const Matrix& A, const Vector& x, Vector& y);

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, CsrTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using IndexType = typename Matrix::index_type;

  const IndexType nnz_per_row = A.nnnz() / A.nrows();

  if (nnz_per_row <= 2) {
    __spmv_csr_vector<2>(A, x, y);
    return;
  }
  if (nnz_per_row <= 4) {
    __spmv_csr_vector<4>(A, x, y);
    return;
  }
  if (nnz_per_row <= 8) {
    __spmv_csr_vector<8>(A, x, y);
    return;
  }
  if (nnz_per_row <= 16) {
    __spmv_csr_vector<16>(A, x, y);
    return;
  }

  __spmv_csr_vector<32>(A, x, y);
}

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, CsrTag, DenseVectorTag, Alg1,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using IndexType = typename Matrix::index_type;
  using ValueType = typename Matrix::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (A.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const IndexType* I = A.row_offsets.data();
  const IndexType* J = A.column_indices.data();
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  Morpheus::Impl::Kernels::spmv_csr_scalar_kernel<IndexType, ValueType>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), I, J, V, x_ptr, y_ptr);
}

template <size_t THREADS_PER_VECTOR, typename Matrix, typename Vector>
void __spmv_csr_vector(const Matrix& A, const Vector& x, Vector& y) {
  using IndexType = typename Matrix::index_type;
  using ValueType = typename Matrix::value_type;

  const IndexType* I = A.row_offsets.data();
  const IndexType* J = A.column_indices.data();
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  const size_t THREADS_PER_BLOCK = 128;
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK,
                                      THREADS_PER_VECTOR>,
      THREADS_PER_BLOCK, (size_t)0);

  const size_t NUM_BLOCKS =
      std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.nrows(), VECTORS_PER_BLOCK));

  Kernels::spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK,
                                  THREADS_PER_VECTOR>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A.nrows(), I, J, V, x_ptr, y_ptr);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP