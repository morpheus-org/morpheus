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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_COPY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_COPY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Copy_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename KeyType, typename SourceType,
          typename DestinationType>
void copy_by_key(
    const KeyType keys, const SourceType& src, DestinationType& dst,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<KeyType> &&
        Morpheus::is_dense_vector_format_container_v<SourceType> &&
        Morpheus::is_dense_vector_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, KeyType, SourceType,
                               DestinationType>>* = nullptr) {
  using size_type  = typename KeyType::size_type;
  using index_type = typename KeyType::value_type;
  using value_type = typename SourceType::value_type;

  if (keys.size() == 0) return;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (keys.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Kernels::copy_by_key_kernel<value_type, index_type, size_type>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(keys.size(), keys.data(), src.data(),
                                      dst.data());

#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("copy_by_key_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_COPY_IMPL_HPP
