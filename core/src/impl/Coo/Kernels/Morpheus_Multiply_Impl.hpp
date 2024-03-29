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

#ifndef MORPHEUS_COO_KERNELS_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_KERNELS_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a
// larger matrix
template <typename SizeType, typename IndexType, typename ValueType>
__global__ void spmv_coo_serial_kernel(const SizeType nnnz, const IndexType* I,
                                       const IndexType* J, const ValueType* V,
                                       const ValueType* x, ValueType* y) {
  for (SizeType n = 0; n < nnnz; n++) {
    y[I[n]] += V[n] * x[J[n]];
  }
}

// spmv_coo_flat_kernel
//
// In this kernel each warp processes an interval of the nonzero values.
// For example, if the matrix contains 128 nonzero values and there are
// two warps and interval_size is 64, then the first warp (warp_id == 0)
// will process the first set of 64 values (interval [0, 64)) and the
// second warp will process // the second set of 64 values
// (interval [64, 128)).  Note that the  number of nonzeros is not always
// a multiple of 32 (the warp size) or 32 * the number of active warps,
// so the last active warp will not always process a "full" interval of
// interval_size.
//
// The first thread in each warp (thread_lane == 0) has a special role:
// it is responsible for keeping track of the "carry" values from one
// iteration to the next.  The carry values consist of the row index and
// partial sum from the previous batch of 32 elements.  In the example
// mentioned before with two warps and 128 nonzero elements, the first
// warp iterates twice and looks at the carry of the first iteration to
// decide whether to include this partial sum into the current batch.
// Specifically, if a row extends over a 32-element boundary, then the
// partial sum is carried over into the new 32-element batch.  If,
// on the other hand, the _last_ row index of the previous batch (the carry)
// differs from the _first_ row index of the current batch (the row
// read by the thread with thread_lane == 0), then the partial sum
// is written out to memory.
//
// Each warp iterates over its interval, processing 32 elements at a time.
// For each batch of 32 elements, the warp does the following
//  1) Fetch the row index, column index, and value for a matrix entry.  These
//     values are loaded from I[n], J[n], and V[n] respectively.
//     The row entry is stored in the shared memory array idx.
//  2) Fetch the corresponding entry from the input vector.  Specifically, for a
//     nonzero entry (i,j) in the matrix, the thread must load the value x[j]
//     from memory.  We use the function fetch_x to control whether the texture
//     cache is used to load the value (UseCache == True) or whether a normal
//     global load is used (UseCache == False).
//  3) The matrix value A(i,j) (which was stored in V[n]) is multiplied by the
//     value x[j] and stored in the shared memory array val.
//  4) The first thread in the warp (thread_lane == 0) considers the "carry"
//     row index and either includes the carried sum in its own sum, or it
//     updates the output vector (y) with the carried sum.
//  5) With row indices in the shared array idx and sums in the shared array
//     val, the warp conducts a segmented scan.  The segmented scan operation
//     looks at the row entries for each thread (stored in idx) to see whether
//     two values belong to the same segment (segments correspond to matrix
//     rows). Consider the following example which consists of 3 segments (note:
//     this example uses a warp size of 16 instead of the usual 32)
//
//           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   # thread_lane
//     idx [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # row indices
//     val [ 4, 6, 5, 0, 8, 3, 2, 8, 3, 1, 4, 9, 2, 5, 2, 4]  # A(i,j) * x(j)
//
//     After the segmented scan the result will be
//
//     val [ 4,10,15,15,23,26, 2,10,13,14, 4,13,15,20,22,26]  # A(i,j) * x(j)
//
//  6) After the warp computes the segmented scan operation
//     each thread except for the last (thread_lane == 31) looks
//     at the row index of the next thread (threadIdx.x + 1) to
//     see if the segment ends here, or continues into the
//     next thread.  The thread at the end of the segment writes
//     the sum into the output vector (y) at the corresponding row
//     index.
//  7) The last thread in each warp (thread_lane == 31) writes
//     its row index and partial sum into the designated spote in the
//     carry_idx and carry_val arrays.  The carry arrays are indexed
//     by warp_lane which is a number in [0, BLOCK_SIZE / 32).
//
//  These steps are repeated until the warp reaches the end of its interval.
//  The carry values at the end of each interval are written to arrays
//  temp_rows and temp_vals, which are processed by a second kernel.
//
template <typename SizeType, typename IndexType, typename ValueType,
          size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) __global__
    void spmv_coo_flat_kernel(const SizeType nnnz, const SizeType interval_size,
                              const IndexType* I, const IndexType* J,
                              const ValueType* V, const ValueType* x,
                              ValueType* y, IndexType* temp_rows,
                              ValueType* temp_vals) {
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
    IndexType row = I[n];            // row index (i)
    ValueType val = V[n] * x[J[n]];  // A(i,j) * x(j)

    if (thread_lane == 0) {
      if (row == rows[idx + LAST_LANE])
        val += ValueType(vals[threadIdx.x + LAST_LANE]);  // row continues
      else
        y[rows[idx + LAST_LANE]] +=
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
      y[row] += ValueType(vals[threadIdx.x]);  // row terminated
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

#endif  // MORPHEUS_COO_KERNELS_MULTIPLY_IMPL_HPP