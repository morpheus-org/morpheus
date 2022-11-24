/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_SERIAL_COPY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_SERIAL_COPY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_FormatTags.hpp>

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
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, KeyType, SourceType,
                               DestinationType>>* = nullptr) {
  MORPHEUS_ASSERT(keys.size() <= src.size(),
                  "Size of keys must be smaller or equal to src size.");
  MORPHEUS_ASSERT(keys.size() <= dst.size(),
                  "Size of keys must be smaller or equal to dst size.");

  for (size_t i = 0; i < keys.size(); i++) {
    dst[i] = src[keys[i]];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DENSEVECTOR_SERIAL_COPY_IMPL_HPP