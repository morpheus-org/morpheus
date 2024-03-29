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

#ifndef MORPHEUS_DENSEMATRIX_COPY_IMPL_HPP
#define MORPHEUS_DENSEMATRIX_COPY_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <Kokkos_Core.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          typename std::enable_if_t<
              Morpheus::is_dense_matrix_format_container_v<SourceType> &&
              Morpheus::is_dense_matrix_format_container_v<DestinationType>>* =
              nullptr) {
  MORPHEUS_ASSERT(
      (dst.nrows() == src.nrows()) && (dst.ncols() == src.ncols()),
      "Destination matrix must have equal shape to the source matrix");

  // Kokkos has src and dst the other way round
  Kokkos::deep_copy(dst.view(), src.const_view());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEMATRIX_COPY_IMPL_HPP