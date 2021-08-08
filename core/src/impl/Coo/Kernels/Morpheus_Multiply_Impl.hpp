/**
 * Morpheus_Multiply_Impl.hpp
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
template <typename IndexType, typename ValueType>
__global__ void spmv_coo_serial_kernel(const IndexType nnnz, const IndexType* I,
                                       const IndexType* J, const ValueType* V,
                                       const ValueType* x, ValueType* y) {
  for (IndexType n = 0; n < nnnz; n++) {
    y[I[n]] += V[n] * x[J[n]];
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif

#endif  // MORPHEUS_COO_KERNELS_MULTIPLY_IMPL_HPP