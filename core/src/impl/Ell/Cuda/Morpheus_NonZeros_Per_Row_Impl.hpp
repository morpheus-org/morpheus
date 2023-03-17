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

#ifndef MORPHEUS_ELL_CUDA_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_ELL_CUDA_NON_ZEROS_PER_ROW_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Ell/Kernels/Morpheus_MatrixAnalytics_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using index_type = typename Matrix::index_type;
  using size_type  = typename Matrix::size_type;
  using value_type = typename Vector::value_type;

  const size_type BLOCK_SIZE = 256;
  const size_type MAX_BLOCKS = max_active_blocks(
      Kernels::count_nnz_per_row_ell_kernel<size_type, index_type, value_type,
                                            BLOCK_SIZE>,
      BLOCK_SIZE, (size_type)0);
  const size_type NUM_BLOCKS = std::min<size_type>(
      MAX_BLOCKS, Impl::ceil_div<size_type>(A.nrows(), BLOCK_SIZE));

  const index_type* J         = A.ccolumn_indices().data();
  value_type* nnz_per_row_ptr = nnz_per_row.data();

  const index_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type pitch               = A.ccolumn_indices().nrows();
  const index_type invalid_index       = A.invalid_index();
  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  Kernels::count_nnz_per_row_ell_kernel<size_type, index_type, value_type,
                                        BLOCK_SIZE>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.ncols(), num_entries_per_row,
                                      pitch, invalid_index, J, nnz_per_row_ptr);

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("count_nnz_per_row_ell_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_ELL_CUDA_NON_ZEROS_PER_ROW_IMPL_HPP