/**
 * Morpheus_Multiply_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
 * Copyright 2008-2014 NVIDIA Corporation
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

#ifndef MORPHEUS_ELL_KERNELS_MULTIPLY_IMPL_HPP
#define MORPHEUS_ELL_KERNELS_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

#include <impl/Morpheus_Utils.hpp>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void spmv_ell_kernel(const SizeType num_rows, const SizeType num_cols,
                         const SizeType num_entries_per_row,
                         const SizeType pitch, const IndexType invalid_index,
                         const IndexType* column_indices,
                         const ValueType* values, const ValueType* x,
                         ValueType* y) {
  const SizeType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const SizeType grid_size = gridDim.x * blockDim.x;

  for (SizeType row = thread_id; row < num_rows; row += grid_size) {
    ValueType sum   = y[row];
    SizeType offset = row;

    for (SizeType n = 0; n < num_entries_per_row; n++) {
      const IndexType col = column_indices[offset];

      if (col != invalid_index) {
        sum += values[offset] * x[col];
      }

      offset += pitch;
    }

    y[row] = sum;
  }
}

}  // namespace Kernels

}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_ELL_KERNELS_MULTIPLY_IMPL_HPP