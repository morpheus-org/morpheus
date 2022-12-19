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

#ifndef MORPHEUS_ELL_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_ELL_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

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
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type = typename SourceType::size_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), src.entries_per_row());

  for (size_type i = 0; i < src.nrows(); i++) {
    for (size_type n = 0; n < src.entries_per_row(); n++) {
      dst.column_indices(i, n) = src.ccolumn_indices(i, n);
      dst.values(i, n)         = src.cvalues(i, n);
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_ell_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;
  using value_type = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  size_type num_entries = 0;

  const size_type num_entries_per_row = src.ccolumn_indices().ncols();
  const index_type invalid_index      = src.invalid_index();

  for (size_type i = 0; i < src.nrows(); i++) {
    for (size_type n = 0; n < num_entries_per_row; n++) {
      const index_type j = src.ccolumn_indices(i, n);
      const value_type v = src.cvalues(i, n);

      if (j != invalid_index) {
        dst.row_indices(num_entries)    = i;
        dst.column_indices(num_entries) = j;
        dst.values(num_entries)         = v;
        num_entries++;
      }
    }
  }

  if (!Impl::is_sorted<ExecSpace>(dst)) {
    Impl::sort_by_row_and_column<ExecSpace>(dst);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_ell_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using src_size_type  = typename SourceType::size_type;
  using src_index_type = typename SourceType::index_type;
  using dst_size_type  = typename DestinationType::size_type;
  using dst_value_type = typename DestinationType::value_type;

  dst_size_type num_entries_per_row = 0;
  dst_size_type entries_ctr         = 0;
  src_index_type row_id             = 0;
  for (src_size_type n = 0; n < src.nnnz(); n++) {
    if (row_id != src.crow_indices(n)) {
      num_entries_per_row =
          entries_ctr > num_entries_per_row ? entries_ctr : num_entries_per_row;
      entries_ctr = 1;  // should be 1;
      row_id      = src.crow_indices(n);
    } else {
      entries_ctr++;
    }
  }

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), num_entries_per_row);

  const dst_size_type nnrows = dst.column_indices().nrows();
  const dst_size_type nncols = dst.column_indices().ncols();
  dst.column_indices().assign(nnrows, nncols, dst.invalid_index());
  dst.values().assign(nnrows, nncols, dst_value_type(0));

  for (src_size_type i = 0; i < src.nnnz(); i++) {
    src_index_type row = src.crow_indices(i);
    src_size_type n    = i % num_entries_per_row;

    dst.column_indices(row, n) = src.ccolumn_indices(i);
    dst.values(row, n)         = src.cvalues(i);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_ELL_SERIAL_CONVERT_IMPL_HPP