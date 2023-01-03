/**
 * Morpheus_MatrixOperations_Impl.hpp
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

#ifndef MORPHEUS_COO_KERNELS_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_COO_KERNELS_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

namespace Kernels {
template <typename ValueType, typename IndexType, typename SizeType>
__global__ void update_coo_diagonal_kernel(const SizeType nnnz,
                                           const IndexType* I,
                                           const IndexType* J, ValueType* V,
                                           const ValueType* diagonal) {
  const SizeType tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nnnz) return;

  if (I[tid] == J[tid]) V[tid] = diagonal[J[tid]];
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA || MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_COO_KERNELS_MATRIXOPERATIONS_IMPL_HPP
