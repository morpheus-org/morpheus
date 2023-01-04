/**
 * Morpheus_VectorAnalytics_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KERNELS_VECTORANALYTICS_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_VECTORANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {

template <typename SizeType, typename IndexType, typename ValueType>
__global__ void count_occurences_dense_vector_serial_kernel(
    const SizeType size, const IndexType* keys, const ValueType* vals,
    ValueType* out) {
  for (SizeType n = 0; n < size; n++) {
    out[keys[n]] += vals[n];
  }
}

template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void count_occurences_dense_vector_flat_kernel(
        const SizeType size, const SizeType interval_size,
        const IndexType* keys, const ValueType* vals, ValueType* out,
        IndexType* temp_keys, ValueType* temp_vals) {
  const SizeType MID_LANE  = WARP_SIZE / 2;
  const SizeType LAST_LANE = WARP_SIZE - 1;

  __shared__ volatile IndexType
      skeys[(WARP_SIZE + MID_LANE) * (BLOCK_SIZE / WARP_SIZE)];
  __shared__ volatile ValueType svals[BLOCK_SIZE];

  const SizeType thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const SizeType thread_lane =
      threadIdx.x & (WARP_SIZE - 1);  // thread index within the warp
  const SizeType warp_id = thread_id / WARP_SIZE;  // global warp index

  const SizeType interval_begin =
      warp_id * interval_size;  // warp's offset into keys,vals
  const SizeType interval_end = Morpheus::Impl::min(
      interval_begin + interval_size, size);  // end of warps's work

  const SizeType idx = MID_LANE * (threadIdx.x / WARP_SIZE + 1) +
                       threadIdx.x;  // thread's index into padded skeys array

  skeys[idx - MID_LANE] = -1;  // fill padding with invalid key index

  if (interval_begin >= interval_end)  // warp has no work to do
    return;

  if (thread_lane == WARP_SIZE - 1) {
    // initialize the carry in values
    skeys[idx]         = keys[interval_begin];
    svals[threadIdx.x] = ValueType(0);
  }

  for (SizeType n = interval_begin + thread_lane; n < interval_end;
       n += WARP_SIZE) {
    IndexType key = keys[n];
    ValueType val = vals[n];

    if (thread_lane == 0) {
      if (key == skeys[idx + LAST_LANE])
        val += ValueType(svals[threadIdx.x + LAST_LANE]);  // key continues
      else
        out[skeys[idx + LAST_LANE]] +=
            ValueType(svals[threadIdx.x + LAST_LANE]);  // key terminated
    }

    skeys[idx]         = key;
    svals[threadIdx.x] = val;

    if (key == skeys[idx - 1]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 1]);
    }
    if (key == skeys[idx - 2]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 2]);
    }
    if (key == skeys[idx - 4]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 4]);
    }
    if (key == skeys[idx - 8]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 8]);
    }
    if (key == skeys[idx - 16]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 16]);
    }

#if defined(MORPHEUS_ENABLE_HIP)
    if (key == skeys[idx - 32]) {
      svals[threadIdx.x] = val += ValueType(svals[threadIdx.x - 32]);
    }
#endif  // MORPHEUS_ENABLE_HIP

    if (thread_lane < LAST_LANE && key != skeys[idx + 1])
      out[key] += ValueType(svals[threadIdx.x]);  // key terminated
  }

  if (thread_lane == LAST_LANE) {
    // write the carry out values
    temp_keys[warp_id] = IndexType(skeys[idx]);
    temp_vals[warp_id] = ValueType(svals[threadIdx.x]);
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_VECTORANALYTICS_IMPL_HPP