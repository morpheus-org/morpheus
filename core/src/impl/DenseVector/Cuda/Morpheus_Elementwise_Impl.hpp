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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_ELEMENTWISE_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_ELEMENTWISE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_Core.hpp>

namespace Morpheus {
namespace Impl {

void elementwise(const DenseVector<double, Kokkos::Cuda>& x,
                 const DenseVector<double, Kokkos::Cuda>& y,
                 DenseVector<double, Kokkos::Cuda>& xy);
// template <typename ValueType, class... P>
// KOKKOS_INLINE_FUNCTION void elementwise(const DenseVector<ValueType, P...>&
// x,
//                                         const DenseVector<ValueType, P...>&
//                                         y, DenseVector<ValueType, P...>&
//                                         xy) {
//   const size_t BLOCK_SIZE = 256;
//   const size_t NUM_BLOCKS = (x.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

//   const ValueType* x_ptr = x.data();
//   const ValueType* y_ptr = y.data();
//   ValueType* xy_ptr      = xy.data();

//   //   Morpheus::Impl::Kernels::elementwise_kernel<ValueType>
//   //       <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(x.size(), x_ptr, y_ptr, xy_ptr);
//   Morpheus::Impl::Kernels::elementwise_kernel<<<NUM_BLOCKS, BLOCK_SIZE,
//   0>>>(
//       x.size(), x_ptr, y_ptr, xy_ptr);
// }

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_ELEMENTWISE_IMPL_HPP