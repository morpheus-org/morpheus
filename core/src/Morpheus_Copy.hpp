/**
 * Morpheus_Copy.hpp
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

#ifndef MORPHEUS_COPY_HPP
#define MORPHEUS_COPY_HPP

#include <impl/Morpheus_Utils.hpp>
#include <impl/Morpheus_Copy_Impl.hpp>
#include <impl/Dynamic/Morpheus_Copy_Impl.hpp>

namespace Morpheus {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst) {
  Morpheus::Impl::copy(src, dst);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type src_begin,
          const typename SourceType::index_type src_end,
          const typename DestinationType::index_type dst_begin,
          const typename DestinationType::index_type dst_end) {
  static_assert(is_dense_vector_format_container_v<SourceType> &&
                    is_dense_vector_format_container_v<DestinationType>,
                "Both src and dst must be vectors.");
  MORPHEUS_ASSERT((src_end - src_begin) == (dst_end - dst_begin),
                  "Source slice range ("
                      << src_begin << ", " << src_end
                      << ") should be equal to the destination slice range ("
                      << dst_begin << ", " << dst_end << ").");

  Morpheus::Impl::copy(src, dst, src_begin, src_end, dst_begin, dst_end);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type begin,
          const typename SourceType::index_type end) {
  Morpheus::copy(src, dst, begin, end, begin, end);
}

template <typename ExecSpace, typename KeyType, typename SourceType,
          typename DestinationType>
void copy_by_key(const KeyType keys, const SourceType& src,
                 DestinationType& dst) {
  Impl::copy_by_key<ExecSpace>(keys, src, dst);
}

}  // namespace Morpheus

#endif  // MORPHEUS_COPY_HPP