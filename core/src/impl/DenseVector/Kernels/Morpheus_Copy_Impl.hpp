/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KERNELS_COPY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_COPY_IMPL_HPP

#if defined(MORPHEUS_ENABLE_HIP)
#include <impl/Morpheus_HIPUtils.hpp>
#elif defined(MORPHEUS_ENABLE_CUDA)
#include <impl/Morpheus_CudaUtils.hpp>
#endif

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename ValueType, typename IndexType, typename SizeType>
__global__ void copy_by_key_kernel(SizeType n, const IndexType* keys,
                                   const ValueType* src, ValueType* dst) {
  const SizeType tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;

  dst[tid] = src[keys[tid]];
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_COPY_IMPL_HPP
