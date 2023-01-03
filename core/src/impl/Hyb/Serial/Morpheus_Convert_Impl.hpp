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

#ifndef MORPHEUS_HYB_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_HYB_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Ell/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Serial/Morpheus_Sort_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_hyb_matrix_format_container_v<SourceType> &&
        Morpheus::is_hyb_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  dst.set_alignment(src.alignment());

  Impl::convert<ExecSpace>(src.cell(), dst.ell());
  Impl::convert<ExecSpace>(src.ccoo(), dst.coo());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_hyb_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  const index_type invalid_index      = src.cell().invalid_index();
  const size_type num_entries_per_row = src.cell().ccolumn_indices().ncols();

  size_type num_entries  = 0;
  size_type coo_progress = 0;

  /* Merge each row of the ELL and COO parts in a single COO row, starting with
   * the ELL part of the row and then moving to the COO part.
   */
  for (size_type i = 0; i < src.nrows(); i++) {
    for (size_type n = 0; n < num_entries_per_row; n++) {
      if (src.cell().ccolumn_indices(i, n) != invalid_index) {
        dst.row_indices(num_entries)    = i;
        dst.column_indices(num_entries) = src.cell().ccolumn_indices(i, n);
        dst.values(num_entries)         = src.cell().cvalues(i, n);
        num_entries++;
      }
    }

    while ((coo_progress < src.ccoo().nnnz()) &&
           ((size_type)src.ccoo().crow_indices(coo_progress) == i)) {
      dst.row_indices(num_entries) = i;
      dst.column_indices(num_entries) =
          src.ccoo().ccolumn_indices(coo_progress);
      dst.values(num_entries) = src.ccoo().cvalues(coo_progress);
      num_entries++;
      coo_progress++;
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
        Morpheus::is_hyb_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;

  const size_type alignment = 32;
  const size_type num_entries_per_row =
      std::max<size_type>(1, src.nnnz() / src.nrows());

  // compute number of nonzeros in the ELL and COO portions
  size_type num_ell_entries = 0;
  size_type entries_ctr     = 0;
  index_type row_id         = 0;
  for (size_type n = 0; n < src.nnnz(); n++) {
    if (row_id != src.crow_indices(n)) {
      entries_ctr = 0;
      row_id      = src.crow_indices(n);
    }

    if (entries_ctr < num_entries_per_row) {
      num_ell_entries++;
    }
    entries_ctr++;
  }

  size_type num_coo_entries = src.nnnz() - num_ell_entries;
  dst.resize(src.nrows(), src.ncols(), num_ell_entries, num_coo_entries,
             num_entries_per_row, alignment);

  const index_type invalid_index = dst.ell().invalid_index();

  dst.ell().column_indices().assign(dst.ell().column_indices().nrows(),
                                    dst.ell().column_indices().ncols(),
                                    invalid_index);
  dst.ell().values().assign(dst.ell().values().nrows(),
                            dst.ell().values().ncols(), 0);

  row_id            = 0;
  size_type coo_nnz = 0;
  for (size_type i = 0, n = 0; i < src.nnnz(); i++) {
    index_type row = src.crow_indices(i);

    if (row_id != src.crow_indices(i)) {
      n      = 0;
      row_id = src.crow_indices(i);
    }

    if ((n < num_entries_per_row) && (row_id == src.crow_indices(i))) {
      dst.ell().column_indices(row, n) = src.ccolumn_indices(i);
      dst.ell().values(row, n)         = src.cvalues(i);
      n++;
    } else if ((row_id == src.crow_indices(i))) {
      dst.coo().row_indices(coo_nnz)    = src.crow_indices(i);
      dst.coo().column_indices(coo_nnz) = src.ccolumn_indices(i);
      dst.coo().values(coo_nnz)         = src.cvalues(i);

      coo_nnz++;
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_HYB_SERIAL_CONVERT_IMPL_HPP