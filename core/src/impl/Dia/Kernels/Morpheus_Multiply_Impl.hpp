/**
 * Morpheus_Multiply_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_DIA_KERNELS_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_KERNELS_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

#include <impl/Morpheus_Utils.hpp>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void spmv_dia_kernel(const SizeType num_rows, const SizeType num_cols,
                         const SizeType num_diagonals, const SizeType pitch,
                         const IndexType* diagonal_offsets,
                         const ValueType* values, const ValueType* x,
                         ValueType* y) {
  __shared__ IndexType offsets[BLOCK_SIZE];

  const SizeType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const SizeType grid_size = BLOCK_SIZE * gridDim.x;

  for (SizeType base = 0; base < num_diagonals; base += BLOCK_SIZE) {
    // read a chunk of the diagonal offsets into shared memory
    const SizeType chunk_size =
        Morpheus::Impl::min(SizeType(BLOCK_SIZE), num_diagonals - base);

    if (threadIdx.x < chunk_size)
      offsets[threadIdx.x] = diagonal_offsets[base + threadIdx.x];

    __syncthreads();

    // process chunk
    for (SizeType row = thread_id; row < num_rows; row += grid_size) {
      ValueType sum = ValueType(0);

      // index into values array
      SizeType idx = row + pitch * base;

      for (SizeType n = 0; n < chunk_size; n++) {
        const IndexType col = row + offsets[n];

        if (col >= 0 && col < (IndexType)num_cols) {
          const ValueType A_ij = values[idx];
          sum += A_ij * x[col];
        }

        idx += pitch;
      }

      y[row] += sum;
    }

    // wait until all threads are done reading offsets
    __syncthreads();
  }
}

}  // namespace Kernels

}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_DIA_KERNELS_MULTIPLY_IMPL_HPP