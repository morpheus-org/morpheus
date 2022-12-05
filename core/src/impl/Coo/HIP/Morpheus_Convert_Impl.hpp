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

#ifndef MORPHEUS_COO_HIP_CONVERT_IMPL_HPP
#define MORPHEUS_COO_HIP_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType&, DestinationType&,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  throw Morpheus::NotImplementedException("convert<Kokkos::Experimental::HIP>");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_COO_HIP_CONVERT_IMPL_HPP