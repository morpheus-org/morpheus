/**
 * Morpheus_NonZeros_Per_Row_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_CSR_HIP_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_CSR_HIP_NON_ZEROS_PER_ROW_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/Csr/Kernels/Morpheus_MatrixAnalytics_Impl.hpp>

namespace Morpheus {
namespace Impl {

// forward decl
template <typename Matrix, typename Vector>
void __count_nnz_per_row_csr_vector(const Matrix& A, Vector& nnz_per_row,
                                    const bool init);

template <typename Matrix, typename Vector>
void __count_nnz_per_row_csr_scalar(const Matrix& A, Vector& nnz_per_row,
                                    const bool init);

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  MORPHEUS_ASSERT(nnz_per_row.size() == A.nrows(),
                  "Destination vector must have equal size to the source "
                  "matrix number of rows");

  switch (A.options()) {
    __count_nnz_per_row_csr_scalar(A, nnz_per_row, init);
    break;
    default: __count_nnz_per_row_csr_vector(A, nnz_per_row, init);
  }
}

template <typename Matrix, typename Vector>
void __count_nnz_per_row_csr_scalar(const Matrix& A, Vector& nnz_per_row,
                                    const bool init) {
  using size_type  = typename Matrix::size_type;
  using index_type = typename Matrix::index_type;
  using value_type = typename Vector::value_type;

  const size_type BLOCK_SIZE = 256;
  const size_type NUM_BLOCKS = Impl::ceil_div<size_type>(A.nrows(), BLOCK_SIZE);

  const index_type* I         = A.crow_offsets().data();
  value_type* nnz_per_row_ptr = nnz_per_row.data();

  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  Morpheus::Impl::Kernels::count_nnz_per_row_csr_scalar_kernel<
      size_type, index_type, value_type>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), I, nnz_per_row_ptr);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError(
      "count_nnz_per_row_csr_scalar_kernel: Kernel execution failed");
#endif
}

template <size_t THREADS_PER_VECTOR, typename Matrix, typename Vector>
void __count_nnz_per_row_csr_vector_dispatch(const Matrix& A,
                                             Vector& nnz_per_row,
                                             const bool init) {
  using size_type  = typename Matrix::size_type;
  using index_type = typename Matrix::index_type;
  using value_type = typename Vector::value_type;

  const index_type* I         = A.crow_offsets().data();
  value_type* nnz_per_row_ptr = nnz_per_row.data();

  const size_type THREADS_PER_BLOCK = 128;
  const size_type VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  const size_type MAX_BLOCKS =
      max_active_blocks(Kernels::count_nnz_per_row_csr_vector_kernel<
                            size_type, index_type, value_type,
                            VECTORS_PER_BLOCK, THREADS_PER_VECTOR>,
                        THREADS_PER_BLOCK, 0);

  const size_type NUM_BLOCKS = std::min<size_type>(
      MAX_BLOCKS, Impl::ceil_div<size_type>(A.nrows(), VECTORS_PER_BLOCK));

  Kernels::count_nnz_per_row_csr_vector_kernel<
      size_type, index_type, value_type, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
      <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0>>>(A.nrows(), I, nnz_per_row_ptr);

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError(
      "count_nnz_per_row_csr_vector_kernel: Kernel execution failed");
#endif
}

template <typename Matrix, typename Vector>
void __count_nnz_per_row_csr_vector(const Matrix& A, Vector& nnz_per_row,
                                    const bool init) {
  using size_type = typename Matrix::size_type;

  const size_type nnz_per_row = A.nnnz() / A.nrows();

  if (nnz_per_row <= 2) {
    __count_nnz_per_row_csr_vector_dispatch<2>(A, nnz_per_row, init);
    return;
  }
  if (nnz_per_row <= 4) {
    __count_nnz_per_row_csr_vector_dispatch<4>(A, nnz_per_row, init);
    return;
  }
  if (nnz_per_row <= 8) {
    __count_nnz_per_row_csr_vector_dispatch<8>(A, nnz_per_row, init);
    return;
  }
  if (nnz_per_row <= 16) {
    __count_nnz_per_row_csr_vector_dispatch<16>(A, nnz_per_row, init);
    return;
  }

  if (nnz_per_row <= 32) {
    __count_nnz_per_row_csr_vector_dispatch<32>(A, nnz_per_row, init);
    return;
  }

  __count_nnz_per_row_csr_vector_dispatch<64>(A, nnz_per_row, init);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_CSR_HIP_NON_ZEROS_PER_ROW_IMPL_HPP