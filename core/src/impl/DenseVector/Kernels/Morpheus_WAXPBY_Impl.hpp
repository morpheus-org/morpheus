/**
 * Morpheus_WAXPBY_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_DENSEVECTOR_KERNELS_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_WAXPBY_IMPL_HPP

#include <cuda.h>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename ValueType, typename IndexType>
__global__ void waxpby_kernel(IndexType n, ValueType alpha, const ValueType* x,
                              ValueType beta, const ValueType* y,
                              ValueType* w) {
  const IndexType tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > n) return;

  if (alpha == 1.0) {
    w[tid] = x[tid] + beta * y[tid];
  } else if (beta == 1.0) {
    w[tid] = alpha * x[tid] + y[tid];
  } else {
    w[tid] = alpha * x[tid] + beta * y[tid];
  }
}
}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_WAXPBY_IMPL_HPP