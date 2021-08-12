/**
 * Morpheus_Multiply_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#ifndef MORPHEUS_CSR_KERNELS_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_KERNELS_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {
// One thread per row
template <typename IndexType, typename ValueType>
__global__ void spmv_csr_scalar_kernel(const IndexType nrows,
                                       const IndexType* Ap, const IndexType* Aj,
                                       const ValueType* Ax, const ValueType* x,
                                       ValueType* y) {
  const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const IndexType grid_size = gridDim.x * blockDim.x;

  for (IndexType row = thread_id; row < nrows; row += grid_size) {
    const IndexType row_start = Ap[row];
    const IndexType row_end   = Ap[row + 1];

    ValueType sum = 0;

    for (IndexType jj = row_start; jj < row_end; jj++)
      sum += Ax[jj] * x[Aj[jj]];

    y[row] = sum;
  }
}

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_kernel
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with
//   the x vector, in parallel.  This division of work implies that
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of
//   work.  Since an entire 32-thread warp is assigned to each row, many
//   threads will remain idle when their row contains a small number
//   of elements.  This code relies on implicit synchronization among
//   threads in a warp.
//
//  Note: THREADS_PER_VECTOR must be one of [2,4,8,16,32]

template <typename IndexType, typename ValueType, size_t VECTORS_PER_BLOCK,
          size_t THREADS_PER_VECTOR>
__global__ void spmv_csr_vector_kernel(const IndexType nrows,
                                       const IndexType* Ap, const IndexType* Aj,
                                       const ValueType* Ax, const ValueType* x,
                                       ValueType* y) {
  __shared__ volatile ValueType
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR +
            THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

  const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

  const IndexType thread_id =
      THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const IndexType thread_lane =
      threadIdx.x & (THREADS_PER_VECTOR - 1);  // thread index within the vector
  const IndexType vector_id =
      thread_id / THREADS_PER_VECTOR;  // global vector index
  const IndexType vector_lane =
      threadIdx.x / THREADS_PER_VECTOR;  // vector index within the block
  const IndexType num_vectors =
      VECTORS_PER_BLOCK * gridDim.x;  // total number of active vectors

  for (IndexType row = vector_id; row < nrows; row += num_vectors) {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if (thread_lane < 2) ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

    const IndexType row_start =
        ptrs[vector_lane][0];  // same as: row_start = Ap[row];
    const IndexType row_end =
        ptrs[vector_lane][1];  // same as: row_end   = Ap[row+1];

    // initialize local sum
    // ValueType sum = (thread_lane == 0) ? initialize(y[row]) : ValueType(0);
    ValueType sum = ValueType(0);

    if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
      // ensure aligned memory access to Aj and Ax

      IndexType jj =
          row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

      // accumulate local sums
      if (jj >= row_start && jj < row_end) sum += Ax[jj] * x[Aj[jj]];

      // accumulate local sums
      for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
        sum += Ax[jj] * x[Aj[jj]];
    } else {
      // accumulate local sums
      for (IndexType jj = row_start + thread_lane; jj < row_end;
           jj += THREADS_PER_VECTOR)
        sum += Ax[jj] * x[Aj[jj]];
    }

    // store local sum in shared memory
    sdata[threadIdx.x] = sum;

    ValueType temp;

    // reduce local sums to row sum
    if (THREADS_PER_VECTOR > 16) {
      temp               = sdata[threadIdx.x + 16];
      sdata[threadIdx.x] = sum += temp;
    }
    if (THREADS_PER_VECTOR > 8) {
      temp               = sdata[threadIdx.x + 8];
      sdata[threadIdx.x] = sum += temp;
    }
    if (THREADS_PER_VECTOR > 4) {
      temp               = sdata[threadIdx.x + 4];
      sdata[threadIdx.x] = sum += temp;
    }
    if (THREADS_PER_VECTOR > 2) {
      temp               = sdata[threadIdx.x + 2];
      sdata[threadIdx.x] = sum += temp;
    }
    if (THREADS_PER_VECTOR > 1) {
      temp               = sdata[threadIdx.x + 1];
      sdata[threadIdx.x] = sum += temp;
    }

    // first thread writes the result
    if (thread_lane == 0) y[row] = ValueType(sdata[threadIdx.x]);
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_CSR_KERNELS_MULTIPLY_IMPL_HPP