/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_CONVERT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <fwd/Morpheus_Fwd_Algorithms.hpp>

namespace Morpheus {

namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, DenseVectorTag, DenseVectorTag,
    typename std::enable_if<
        std::is_same<typename SourceType::memory_space,
                     typename DestinationType::memory_space>::value &&
        is_HostSpace_v<typename SourceType::memory_space>>::type* = nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.size());

  for (index_type i = 0; i < src.size(); i++) {
    dst[i] = src[i];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_CONVERT_IMPL_HPP