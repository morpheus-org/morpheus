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

#ifndef MORPHEUS_DENSEVECTOR_COPY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_COPY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/DenseVector/Serial/Morpheus_Copy_Impl.hpp>
#include <impl/DenseVector/OpenMP/Morpheus_Copy_Impl.hpp>
#include <impl/DenseVector/Cuda/Morpheus_Copy_Impl.hpp>

#include <Kokkos_Core.hpp>

#include <utility>  // std::pair

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          typename std::enable_if_t<
              Morpheus::is_dense_vector_format_container_v<SourceType> &&
              Morpheus::is_dense_vector_format_container_v<DestinationType>>* =
              nullptr) {
  MORPHEUS_ASSERT(
      dst.size() == src.size(),
      "Destination vector must be of equal size to the source vector");

  // Kokkos has src and dst the other way round
  Kokkos::deep_copy(dst.view(), src.const_view());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type src_begin,
          const typename SourceType::index_type src_end,
          const typename DestinationType::index_type dst_begin,
          const typename DestinationType::index_type dst_end,
          typename std::enable_if_t<
              Morpheus::is_dense_vector_format_container_v<SourceType> &&
              Morpheus::is_dense_vector_format_container_v<DestinationType>>* =
              nullptr) {
  auto src_sub =
      Kokkos::subview(src.const_view(), std::make_pair(src_begin, src_end));
  auto dst_sub =
      Kokkos::subview(dst.view(), std::make_pair(dst_begin, dst_end));

  // Kokkos has src and dst the other way round
  Kokkos::deep_copy(dst_sub, src_sub);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_COPY_IMPL_HPP