/**
 * Morpheus_Segmented_Reduction_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KERNELS_SEGMENTED_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_SEGMENTED_REDUCTION_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {

// forward decl
template <typename IndexType, typename ValueType>
__device__ void segreduce_block(const IndexType* idx, ValueType* val);

template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void reduce_update_kernel(const SizeType num_warps,
                              const IndexType* temp_rows,
                              const ValueType* temp_vals, ValueType* y) {
  __shared__ IndexType rows[BLOCK_SIZE + 1];
  __shared__ ValueType vals[BLOCK_SIZE + 1];

  const SizeType end = num_warps - (num_warps & (BLOCK_SIZE - 1));

  if (threadIdx.x == 0) {
    rows[BLOCK_SIZE] = (IndexType)-1;
    vals[BLOCK_SIZE] = (ValueType)0;
  }

  __syncthreads();

  SizeType i = threadIdx.x;

  while (i < end) {
    // do full blocks
    rows[threadIdx.x] = temp_rows[i];
    vals[threadIdx.x] = temp_vals[i];

    __syncthreads();

    segreduce_block(rows, vals);

    if (rows[threadIdx.x] != rows[threadIdx.x + 1])
      y[rows[threadIdx.x]] += vals[threadIdx.x];

    __syncthreads();

    i += BLOCK_SIZE;
  }

  if (end < num_warps) {
    if (i < num_warps) {
      rows[threadIdx.x] = temp_rows[i];
      vals[threadIdx.x] = temp_vals[i];
    } else {
      rows[threadIdx.x] = (IndexType)-1;
      vals[threadIdx.x] = (ValueType)0;
    }

    __syncthreads();

    segreduce_block(rows, vals);

    if (i < num_warps)
      if (rows[threadIdx.x] != rows[threadIdx.x + 1])
        y[rows[threadIdx.x]] += vals[threadIdx.x];
  }
}

template <typename IndexType, typename ValueType>
__device__ void segreduce_block(const IndexType* idx, ValueType* val) {
  ValueType left = 0;
  if (threadIdx.x >= 1 && idx[threadIdx.x] == idx[threadIdx.x - 1]) {
    left = val[threadIdx.x - 1];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 2 && idx[threadIdx.x] == idx[threadIdx.x - 2]) {
    left = val[threadIdx.x - 2];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 4 && idx[threadIdx.x] == idx[threadIdx.x - 4]) {
    left = val[threadIdx.x - 4];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 8 && idx[threadIdx.x] == idx[threadIdx.x - 8]) {
    left = val[threadIdx.x - 8];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 16 && idx[threadIdx.x] == idx[threadIdx.x - 16]) {
    left = val[threadIdx.x - 16];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 32 && idx[threadIdx.x] == idx[threadIdx.x - 32]) {
    left = val[threadIdx.x - 32];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 64 && idx[threadIdx.x] == idx[threadIdx.x - 64]) {
    left = val[threadIdx.x - 64];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128]) {
    left = val[threadIdx.x - 128];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
  if (threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256]) {
    left = val[threadIdx.x - 256];
  }
  __syncthreads();
  val[threadIdx.x] += left;
  left = 0;
  __syncthreads();
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_SEGMENTED_REDUCTION_IMPL_HPP