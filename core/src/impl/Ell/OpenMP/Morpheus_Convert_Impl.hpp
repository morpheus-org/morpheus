/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_ELL_OPENMP_CONVERT_IMPL_HPP
#define MORPHEUS_ELL_OPENMP_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Coo/Serial/Morpheus_Sort_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_ell_matrix_format_container_v<SourceType> &&
        Morpheus::is_ell_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type = typename SourceType::size_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), src.entries_per_row());
  dst.column_indices().assign(dst.column_indices().nrows(),
                              dst.column_indices().ncols(),
                              dst.invalid_index());
  dst.values().assign(dst.values().nrows(), dst.values().ncols(), 0);

#pragma omp parallel for collapse(2)
  for (size_type i = 0; i < src.cvalues().nrows(); i++) {
    for (size_type j = 0; j < src.cvalues().ncols(); j++) {
      dst.column_indices(i, j) = src.ccolumn_indices(i, j);
      dst.values(i, j)         = src.cvalues(i, j);
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType&, DestinationType&,
    typename std::enable_if<
        Morpheus::is_ell_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  throw Morpheus::NotImplementedException("convert<Morpheus::OpenMP>");
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType&, DestinationType&,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_ell_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  throw Morpheus::NotImplementedException("convert<Morpheus::OpenMP>");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ELL_OPENMP_CONVERT_IMPL_HPP