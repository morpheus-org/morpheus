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

#ifndef MORPHEUS_COO_HIP_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_COO_HIP_NON_ZEROS_PER_ROW_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/Coo/Kernels/Morpheus_MatrixAnalytics_Impl.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Segmented_Reduction_Impl.hpp>

namespace Morpheus {
namespace Impl {

// forward decl
template <typename Matrix, typename Vector>
void __count_nnz_per_row_coo_flat(const Matrix& A, Vector& nnz_per_row,
                                  const bool init);

template <typename Matrix, typename Vector>
void __count_nnz_per_row_coo_serial(const Matrix& A, Vector& nnz_per_row,
                                    const bool init);

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  switch (A.options()) {
    case MATOPT_SHORT_ROWS:
      __count_nnz_per_row_coo_serial(A, nnz_per_row, init);
      break;
    default: __count_nnz_per_row_coo_flat(A, nnz_per_row, init);
  }
}

template <typename Matrix, typename Vector>
void __count_nnz_per_row_coo_serial(const Matrix& A, Vector& nnz_per_row,
                                    const bool init) {
  using size_type             = typename Matrix::size_type;
  using index_type            = typename Matrix::index_type;
  using value_type            = typename Vector::value_type;
  const index_type* I         = A.crow_indices().data();
  value_type* nnz_per_row_ptr = nnz_per_row.data();

  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  Kernels::count_nnz_per_row_coo_serial_kernel<size_type, index_type,
                                               value_type>
      <<<1, 1>>>(A.nnnz(), I, nnz_per_row_ptr);

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError(
      "count_nnz_per_row_coo_serial_kernel: Kernel execution failed");
#endif
}

template <typename Matrix, typename Vector>
void __count_nnz_per_row_coo_flat(const Matrix& A, Vector& nnz_per_row,
                                  const bool init) {
  using size_type  = typename Matrix::size_type;
  using index_type = typename Matrix::index_type;
  using value_type = typename Vector::value_type;

  MORPHEUS_ASSERT(nnz_per_row.size() == A.nrows(),
                  "Destination vector must have equal size to the source "
                  "matrix number of rows");

  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  if (A.nnnz() == 0) {
    // empty matrix
    return;
  } else if (A.nnnz() < static_cast<size_type>(WARP_SIZE)) {
    // small matrix
    Kernels::count_nnz_per_row_coo_serial_kernel<size_type, index_type,
                                                 value_type>
        <<<1, 1, 0>>>(A.nnnz(), A.crow_indices().data(), nnz_per_row.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
    getLastHIPError("spmv_coo_serial_kernel: Kernel execution failed");
#endif
    return;
  }

  const size_type BLOCK_SIZE = 256;
  const size_type MAX_BLOCKS = max_active_blocks(
      Kernels::count_nnz_per_row_coo_flat_kernel<size_type, index_type,
                                                 value_type, BLOCK_SIZE>,
      BLOCK_SIZE, 0);
  const size_type WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

  const size_type num_units = A.nnnz() / WARP_SIZE;
  const size_type num_warps = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
  const size_type num_blocks =
      Impl::ceil_div<size_type>(num_warps, WARPS_PER_BLOCK);
  const size_type num_iters = Impl::ceil_div<size_type>(num_units, num_warps);

  const size_type interval_size = WARP_SIZE * num_iters;

  const size_type tail =
      num_units * WARP_SIZE;  // do the last few nonzeros separately (fewer
                              // than WARP_SIZE elements)

  const size_type active_warps =
      (interval_size == 0) ? 0 : Impl::ceil_div<size_type>(tail, interval_size);

  typename Matrix::index_array_type temp_rows(active_warps, 0);
  Vector temp_vals(active_warps, 0);

  Kernels::count_nnz_per_row_coo_flat_kernel<size_type, index_type, value_type,
                                             BLOCK_SIZE>
      <<<num_blocks, BLOCK_SIZE, 0>>>(
          tail, interval_size, A.crow_indices().data(), nnz_per_row.data(),
          temp_rows.data(), temp_vals.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("count_nnz_per_row_coo_flat_kernel: Kernel execution failed");
#endif

  Kernels::reduce_update_kernel<size_type, index_type, value_type, BLOCK_SIZE>
      <<<1, BLOCK_SIZE, 0>>>(active_warps, temp_rows.data(), temp_vals.data(),
                             nnz_per_row.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("reduce_update_kernel: Kernel execution failed");
#endif

  Kernels::count_nnz_per_row_coo_serial_kernel<size_type, index_type,
                                               value_type><<<1, 1, 0>>>(
      A.nnnz() - tail, A.crow_indices().data() + tail, nnz_per_row.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError(
      "count_nnz_per_row_coo_serial_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_COO_HIP_NON_ZEROS_PER_ROW_IMPL_HPP