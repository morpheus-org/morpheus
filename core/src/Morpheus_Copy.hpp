/**
 * Morpheus_Copy.hpp
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

#ifndef MORPHEUS_COPY_HPP
#define MORPHEUS_COPY_HPP

#include <impl/Morpheus_Copy_Impl.hpp>

namespace Morpheus {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst) {
  Impl::copy(src, dst, typename SourceType::tag(),
             typename DestinationType::tag());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type src_begin,
          const typename SourceType::index_type src_end,
          const typename DestinationType::index_type dst_begin,
          const typename DestinationType::index_type dst_end) {
  static_assert(is_vector_v<typename SourceType::tag> &&
                    is_vector_v<typename DestinationType::tag>,
                "Both src and dst must be vectors.");
  assert((src_end - src_begin) != (dst_end - dst_begin));

  Impl::copy(src, dst, src_begin, src_end, dst_begin, dst_end,
             typename SourceType::tag(), typename DestinationType::tag());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type begin,
          const typename SourceType::index_type end) {
  Morpheus::copy(src, dst, begin, end, begin, end);
}

}  // namespace Morpheus

#endif  // MORPHEUS_COPY_HPP