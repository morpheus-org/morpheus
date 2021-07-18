/**
 * Morpheus_Multiply_Kernels.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#ifndef MORPHEUS_CUDA_MULTIPLY_KERNELS_HPP
#define MORPHEUS_CUDA_MULTIPLY_KERNELS_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

namespace Morpheus {
namespace Impl {
namespace Kernels {

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a
// larger matrix

template <typename IndexType, typename ValueType>
MORPHEUS_INLINE_FUNCTION void spmv_coo_serial_kernel(
    const IndexType nnnz, const IndexType* I, const IndexType* J,
    const ValueType* V, const ValueType* x, ValueType* y) {
  for (IndexType n = 0; n < nnnz; n++) {
    y[I[n]] += V[n] * x[J[n]];
  }
}

template <typename IndexType, typename ValueType>
MORPHEUS_INLINE_FUNCTION void spmv_csr_scalar_kernel(
    const IndexType nrows, const IndexType* Ap, const IndexType* Aj,
    const ValueType* Ax, const ValueType* x, ValueType* y) {
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

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CUDA_MULTIPLY_KERNELS_HPP