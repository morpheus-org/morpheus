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

#ifndef MORPHEUS_COO_OPENMP_CONVERT_IMPL_HPP
#define MORPHEUS_COO_OPENMP_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>::type* = nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

// element-wise copy of indices and values
#pragma omp parallel for
  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.row_indices(n) = src.crow_indices(n);
  }

#pragma omp parallel for
  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.column_indices(n) = src.ccolumn_indices(n);
  }

#pragma omp parallel for
  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.values(n) = src.cvalues(n);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_CONVERT_IMPL_HPP