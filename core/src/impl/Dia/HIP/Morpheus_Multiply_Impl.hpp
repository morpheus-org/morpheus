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

#ifndef MORPHEUS_DIA_HIP_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_HIP_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/Dia/Kernels/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::spmv_dia_kernel<index_type, value_type, BLOCK_SIZE>, BLOCK_SIZE,
      (size_t)sizeof(index_type) * BLOCK_SIZE);
  const size_t NUM_BLOCKS =
      std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.nrows(), BLOCK_SIZE));

  const index_type* D     = A.cdiagonal_offsets().data();
  const value_type* V     = A.cvalues().data();
  const value_type* x_ptr = x.data();
  value_type* y_ptr       = y.data();

  const index_type num_diagonals = A.cvalues().ncols();
  const index_type pitch         = A.cvalues().nrows();

  if (num_diagonals == 0) {
    // empty matrix
    return;
  }

  if (init) {
    y.assign(y.size(), 0);
  }

  Kernels::spmv_dia_kernel<index_type, value_type, BLOCK_SIZE>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.ncols(), num_diagonals,
                                      pitch, D, V, x_ptr, y_ptr);

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("spmv_dia_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DIA_HIP_MULTIPLY_IMPL_HPP