/**
 * Morpheus_MatrixAnalytics_Impl.hpp
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

#ifndef MORPHEUS_CSR_KERNELS_MATRIXANALYTICS_IMPL_HPP
#define MORPHEUS_CSR_KERNELS_MATRIXANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {
// One thread per row
template <typename SizeType, typename IndexType, typename ValueType>
__global__ void count_nnz_per_row_csr_scalar_kernel(const SizeType nrows,
                                                    const IndexType* Ap,
                                                    ValueType* nnz_per_row) {
  const SizeType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const SizeType grid_size = gridDim.x * blockDim.x;

  for (SizeType row = thread_id; row < nrows; row += grid_size) {
    const IndexType row_start = Ap[row];
    const IndexType row_end   = Ap[row + 1];

    ValueType non_zeros = ValueType(0);

    for (IndexType jj = row_start; jj < row_end; jj++) non_zeros++;

    nnz_per_row[row] += non_zeros;
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

template <typename SizeType, typename IndexType, typename ValueType,
          size_t VECTORS_PER_BLOCK, size_t THREADS_PER_VECTOR>
__global__ void count_nnz_per_row_csr_vector_kernel(const SizeType nrows,
                                                    const IndexType* Ap,
                                                    ValueType* nnz_per_row) {
  __shared__ volatile ValueType
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR +
            THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

  const SizeType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

  const SizeType thread_id =
      THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const SizeType thread_lane =
      threadIdx.x & (THREADS_PER_VECTOR - 1);  // thread index within the vector
  const SizeType vector_id =
      thread_id / THREADS_PER_VECTOR;  // global vector index
  const SizeType vector_lane =
      threadIdx.x / THREADS_PER_VECTOR;  // vector index within the block
  const SizeType num_vectors =
      VECTORS_PER_BLOCK * gridDim.x;  // total number of active vectors

  for (SizeType row = vector_id; row < nrows; row += num_vectors) {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if (thread_lane < 2) ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

    const IndexType row_start =
        ptrs[vector_lane][0];  // same as: row_start = Ap[row];
    const IndexType row_end =
        ptrs[vector_lane][1];  // same as: row_end   = Ap[row+1];

    // initialize local sum
    ValueType non_zeros = ValueType(0);

    if (THREADS_PER_VECTOR == WARP_SIZE && row_end - row_start > WARP_SIZE) {
      // ensure aligned memory access to Aj and Ax

      IndexType jj =
          row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

      // accumulate local non_zeros
      if (jj >= row_start && jj < row_end) non_zeros++;

      // accumulate local non_zeros
      for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
        non_zeros++;
    } else {
      // accumulate local non_zeros
      for (IndexType jj = row_start + thread_lane; jj < row_end;
           jj += THREADS_PER_VECTOR)
        non_zeros++;
    }

    // store local sum in shared memory
    sdata[threadIdx.x] = non_zeros;

    ValueType temp;

#if defined(MORPHEUS_ENABLE_HIP)
    if (THREADS_PER_VECTOR > 32) {
      temp               = sdata[threadIdx.x + 32];
      sdata[threadIdx.x] = non_zeros += temp;
    }
#endif  // MORPHEUS_ENABLE_HIP

    // reduce local non_zeros to row non_zeros
    if (THREADS_PER_VECTOR > 16) {
      temp               = sdata[threadIdx.x + 16];
      sdata[threadIdx.x] = non_zeros += temp;
    }
    if (THREADS_PER_VECTOR > 8) {
      temp               = sdata[threadIdx.x + 8];
      sdata[threadIdx.x] = non_zeros += temp;
    }
    if (THREADS_PER_VECTOR > 4) {
      temp               = sdata[threadIdx.x + 4];
      sdata[threadIdx.x] = non_zeros += temp;
    }
    if (THREADS_PER_VECTOR > 2) {
      temp               = sdata[threadIdx.x + 2];
      sdata[threadIdx.x] = non_zeros += temp;
    }
    if (THREADS_PER_VECTOR > 1) {
      temp               = sdata[threadIdx.x + 1];
      sdata[threadIdx.x] = non_zeros += temp;
    }

    // first thread writes the result
    if (thread_lane == 0) nnz_per_row[row] += ValueType(sdata[threadIdx.x]);
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_CSR_KERNELS_MATRIXANALYTICS_IMPL_HPP