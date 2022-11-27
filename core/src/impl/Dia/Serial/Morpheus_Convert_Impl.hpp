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

#ifndef MORPHEUS_DIA_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_DIA_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

// TODO: Remove use of set during Coo to Dia Conversion
#include <set>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dia_matrix_format_container_v<SourceType> &&
        Morpheus::is_dia_matrix_format_container_v<DestinationType> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz(),
             src.cdiagonal_offsets().size());

  // element-wise copy of indices and values
  for (index_type n = 0; n < src.cdiagonal_offsets().size(); n++) {
    dst.diagonal_offsets(n) = src.cdiagonal_offsets(n);
  }

  for (index_type j = 0; j < src.cvalues().ncols(); j++) {
    for (index_type i = 0; i < src.cvalues().nrows(); i++) {
      dst.values(i, j) = src.cvalues(i, j);
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dia_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;
  using value_type = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  const index_type ndiag = src.cvalues().ncols();

  for (index_type i = 0, nnzid = 0; i < ndiag; i++) {
    const index_type k = src.cdiagonal_offsets(i);

    const index_type i_start = std::max<index_type>(0, -k);
    const index_type j_start = std::max<index_type>(0, k);

    // number of elements to process in this diagonal
    const index_type N = std::min(src.nrows() - i_start, src.ncols() - j_start);

    for (index_type n = 0; n < N; n++) {
      const value_type temp = src.cvalues(i_start + n, i);

      if (temp != value_type(0)) {
        dst.row_indices(nnzid)    = i_start + n;
        dst.column_indices(nnzid) = j_start + n;
        dst.values(nnzid)         = temp;
        nnzid                     = nnzid + 1;
      }
    }
  }

  if (!dst.is_sorted()) {
    dst.sort();
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_dia_matrix_format_container_v<DestinationType> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;

  if (src.nnnz() == 0) {
    dst.resize(src.nrows(), src.ncols(), src.nnnz(), 0);
    return;
  }

  std::vector<index_type> diag_map(src.nnnz(), 0);

  // Find on which diagonal each entry sits on
  for (index_type n = 0; n < index_type(src.nnnz()); n++) {
    diag_map[n] = src.ccolumn_indices(n) - src.crow_indices(n);
  }

  // Create unique diagonal set
  std::set<index_type> diag_set(diag_map.begin(), diag_map.end());
  index_type ndiags = index_type(diag_set.size());

  if (Impl::exceeds_tolerance(src.nrows(), src.nnnz(), ndiags)) {
    throw Morpheus::FormatConversionException(
        "DiaMatrix fill-in would exceed maximum tolerance");
  }

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), ndiags);

  for (auto it = diag_set.cbegin(); it != diag_set.cend(); ++it) {
    auto i                  = std::distance(diag_set.cbegin(), it);
    dst.diagonal_offsets(i) = *it;
  }

  for (index_type n = 0; n < index_type(src.nnnz()); n++) {
    for (index_type i = 0; i < index_type(ndiags); i++) {
      if (diag_map[n] == dst.diagonal_offsets(i)) {
        dst.values(src.crow_indices(n), i) = src.cvalues(n);
        break;
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DIA_SERIAL_CONVERT_IMPL_HPP