/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Reduction.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Cuda/Morpheus_Workspace.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Dot_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2>
typename Vector1::value_type dot(
    const typename Vector1::index_type n, const Vector1& x, const Vector2& y,
    DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector1,
                               Vector2>>* = nullptr) {
  using index_type = typename Vector1::index_type;
  using value_type = typename Vector1::non_const_value_type;

//   const size_t BLOCK_SIZE = 256;
//   const size_t NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudotspace.allocate<value_type>(n);

  Kernels::dot_kernel_part1<256, value_type, index_type>
      <<<256, 256>>>(n, x.data(), y.data(), cudotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("dot: Kernel execution failed");
#endif

  Kernels::dot_kernel_part2<256, value_type><<<1, 256>>>(cudotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("dot: Kernel execution failed");
#endif

  value_type local_result;
  cudaMemcpy(&local_result, cudotspace.data<value_type>(), sizeof(value_type), cudaMemcpyDeviceToHost);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("dot: Kernel execution failed");
#endif

  return local_result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP
