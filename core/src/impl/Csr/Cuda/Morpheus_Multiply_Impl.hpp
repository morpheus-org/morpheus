/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Csr/Kernels/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

// forward decl
template <typename Matrix, typename Vector>
void __spmv_csr_vector(const Matrix& A, const Vector& x, Vector& y,
                       const bool init);

template <typename Matrix, typename Vector>
void __spmv_csr_scalar(const Matrix& A, const Vector& x, Vector& y,
                       const bool init);

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  switch (A.options()) {
    case MATOPT_SHORT_ROWS: __spmv_csr_scalar(A, x, y, init); break;
    default: __spmv_csr_vector(A, x, y, init);
  }
}

template <typename Matrix, typename Vector>
void __spmv_csr_scalar(const Matrix& A, const Vector& x, Vector& y,
                       const bool init) {
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (A.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const index_type* I = A.crow_offsets().data();
  const index_type* J = A.ccolumn_indices().data();
  const value_type* V = A.cvalues().data();

  const value_type* x_ptr = x.data();
  value_type* y_ptr       = y.data();

  if (init) {
    y.assign(y.size(), 0);
  }

  Morpheus::Impl::Kernels::spmv_csr_scalar_kernel<index_type, value_type>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), I, J, V, x_ptr, y_ptr);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("spmv_csr_scalar_kernel: Kernel execution failed");
#endif
}

template <size_t THREADS_PER_VECTOR, typename Matrix, typename Vector>
void __spmv_csr_vector_dispatch(const Matrix& A, const Vector& x, Vector& y,
                                const bool init) {
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const index_type* I = A.crow_offsets().data();
  const index_type* J = A.ccolumn_indices().data();
  const value_type* V = A.cvalues().data();

  const value_type* x_ptr = x.data();
  value_type* y_ptr       = y.data();

  const size_t THREADS_PER_BLOCK = 128;
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

  if (init) {
    y.assign(y.size(), 0);
  }

  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::spmv_csr_vector_kernel<index_type, value_type, VECTORS_PER_BLOCK,
                                      THREADS_PER_VECTOR>,
      THREADS_PER_BLOCK, (size_t)0);

  const size_t NUM_BLOCKS =
      std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.nrows(), VECTORS_PER_BLOCK));

  Kernels::spmv_csr_vector_kernel<index_type, value_type, VECTORS_PER_BLOCK,
                                  THREADS_PER_VECTOR>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A.nrows(), I, J, V, x_ptr, y_ptr);

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("spmv_csr_vector_kernel: Kernel execution failed");
#endif
}

template <typename Matrix, typename Vector>
void __spmv_csr_vector(const Matrix& A, const Vector& x, Vector& y,
                       const bool init) {
  using index_type = typename Matrix::index_type;

  const index_type nnz_per_row = A.nnnz() / A.nrows();

  if (nnz_per_row <= 2) {
    __spmv_csr_vector_dispatch<2>(A, x, y, init);
    return;
  }
  if (nnz_per_row <= 4) {
    __spmv_csr_vector_dispatch<4>(A, x, y, init);
    return;
  }
  if (nnz_per_row <= 8) {
    __spmv_csr_vector_dispatch<8>(A, x, y, init);
    return;
  }
  if (nnz_per_row <= 16) {
    __spmv_csr_vector_dispatch<16>(A, x, y, init);
    return;
  }

  __spmv_csr_vector_dispatch<32>(A, x, y, init);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_CSR_CUDA_MULTIPLY_IMPL_HPP