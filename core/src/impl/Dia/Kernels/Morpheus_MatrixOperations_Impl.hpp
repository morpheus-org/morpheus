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

#ifndef MORPHEUS_DIA_KERNELS_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DIA_KERNELS_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {
template <typename ValueType, typename IndexType, typename SizeType>
__global__ void update_dia_diagonal_kernel(const SizeType num_rows,
                                           const SizeType num_cols,
                                           const SizeType num_diagonals,
                                           const SizeType pitch,
                                           const IndexType* diagonal_offsets,
                                           ValueType* values,
                                           const ValueType* diagonal) {
  const SizeType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const SizeType grid_size = gridDim.x * blockDim.x;

  for (SizeType row = thread_id; row < num_rows; row += grid_size) {
    for (SizeType n = 0; n < num_diagonals; n++) {
      const IndexType col = row + diagonal_offsets[n];
      const SizeType idx  = row + pitch * n;

      if ((col >= 0 && col < (IndexType)num_cols) && (col == (IndexType)row)) {
        values[idx] = (values[idx] == ValueType(0)) ? 0 : diagonal[col];
      }
    }
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA || MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DIA_KERNELS_MATRIXOPERATIONS_IMPL_HPP