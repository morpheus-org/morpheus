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
          const typename SourceType::index_type begin,
          const typename SourceType::index_type end) {
  Impl::copy(src, dst, begin, end, typename SourceType::tag(),
             typename DestinationType::tag());
}
}  // namespace Morpheus

#endif  // MORPHEUS_COPY_HPP