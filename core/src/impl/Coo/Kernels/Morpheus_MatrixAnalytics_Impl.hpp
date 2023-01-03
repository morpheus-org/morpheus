/**
 * Morpheus_MatrixAnalytics_Impl.hpp
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

#ifndef MORPHEUS_COO_KERNELS_MATRIXANALYTICS_IMPL_HPP
#define MORPHEUS_COO_KERNELS_MATRIXANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {

template <typename SizeType, typename IndexType, typename ValueType>
__global__ void count_nnz_per_row_coo_serial_kernel(const SizeType nnnz,
                                                    const IndexType* I,
                                                    ValueType* nnz_per_row) {
  for (SizeType n = 0; n < nnnz; n++) {
    nnz_per_row[I[n]]++;
  }
}

template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void count_nnz_per_row_coo_flat_kernel(
        const SizeType nnnz, const SizeType interval_size, const IndexType* I,
        ValueType* nnz_per_row, IndexType* temp_rows, ValueType* temp_vals) {
  const SizeType MID_LANE  = WARP_SIZE / 2;
  const SizeType LAST_LANE = WARP_SIZE - 1;

  __shared__ volatile IndexType
      rows[(WARP_SIZE + MID_LANE) * (BLOCK_SIZE / WARP_SIZE)];
  __shared__ volatile ValueType vals[BLOCK_SIZE];

  const SizeType thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const SizeType thread_lane =
      threadIdx.x & (WARP_SIZE - 1);  // thread index within the warp
  const SizeType warp_id = thread_id / WARP_SIZE;  // global warp index

  const SizeType interval_begin =
      warp_id * interval_size;  // warp's offset into I,J,V
  const SizeType interval_end = Morpheus::Impl::min(
      interval_begin + interval_size, nnnz);  // end of warps's work

  const SizeType idx = MID_LANE * (threadIdx.x / WARP_SIZE + 1) +
                       threadIdx.x;  // thread's index into padded rows array

  rows[idx - MID_LANE] = -1;  // fill padding with invalid row index

  if (interval_begin >= interval_end)  // warp has no work to do
    return;

  if (thread_lane == WARP_SIZE - 1) {
    // initialize the carry in values
    rows[idx]         = I[interval_begin];
    vals[threadIdx.x] = ValueType(0);
  }

  for (SizeType n = interval_begin + thread_lane; n < interval_end;
       n += WARP_SIZE) {
    IndexType row = I[n];  // row index (i)
    ValueType val = 1;

    if (thread_lane == 0) {
      if (row == rows[idx + LAST_LANE])
        val += ValueType(vals[threadIdx.x + LAST_LANE]);  // row continues
      else
        nnz_per_row[rows[idx + LAST_LANE]] +=
            ValueType(vals[threadIdx.x + LAST_LANE]);  // row terminated
    }

    rows[idx]         = row;
    vals[threadIdx.x] = val;

    if (row == rows[idx - 1]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 1]);
    }
    if (row == rows[idx - 2]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 2]);
    }
    if (row == rows[idx - 4]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 4]);
    }
    if (row == rows[idx - 8]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 8]);
    }
    if (row == rows[idx - 16]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 16]);
    }

#if defined(MORPHEUS_ENABLE_HIP)
    if (row == rows[idx - 32]) {
      vals[threadIdx.x] = val += ValueType(vals[threadIdx.x - 32]);
    }
#endif  // MORPHEUS_ENABLE_HIP

    if (thread_lane < LAST_LANE && row != rows[idx + 1])
      nnz_per_row[row] += ValueType(vals[threadIdx.x]);  // row terminated
  }

  if (thread_lane == LAST_LANE) {
    // write the carry out values
    temp_rows[warp_id] = IndexType(rows[idx]);
    temp_vals[warp_id] = ValueType(vals[threadIdx.x]);
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_COO_KERNELS_MATRIXANALYTICS_IMPL_HPP