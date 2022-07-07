/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_CSR_KOKKOS_CONVERT_IMPL_HPP
#define MORPHEUS_CSR_KOKKOS_CONVERT_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>

#include <Morpheus_Exceptions.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
inline void convert(
    const SourceType& src, DestinationType& dst, CsrTag, CsrTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>* = nullptr) {
  throw Morpheus::NotImplementedException("convert<Kokkos>");
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
inline void convert(
    const SourceType& src, DestinationType& dst, CsrTag, CooTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>* = nullptr) {
  throw Morpheus::NotImplementedException("convert<Kokkos>");
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
inline void convert(
    const SourceType& src, DestinationType& dst, CooTag, CsrTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>* = nullptr) {
  throw Morpheus::NotImplementedException("convert<Kokkos>");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_CONVERT_IMPL_HPP