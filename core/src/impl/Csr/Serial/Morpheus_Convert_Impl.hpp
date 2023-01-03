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

#ifndef MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_csr_matrix_format_container_v<SourceType> &&
        Morpheus::is_csr_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type = typename SourceType::size_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // element-wise copy of indices and values
  for (size_type n = 0; n < src.nrows() + 1; n++) {
    dst.row_offsets(n) = src.crow_offsets(n);
  }

  for (size_type n = 0; n < src.nnnz(); n++) {
    dst.column_indices(n) = src.ccolumn_indices(n);
  }

  for (size_type n = 0; n < src.nnnz(); n++) {
    dst.values(n) = src.cvalues(n);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_csr_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  // Complexity: Linear.  Specifically O(nnz(csr) + max(n_row,n_col))
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // expand compressed indices
  for (size_type i = 0; i < src.nrows(); i++) {
    for (index_type jj = src.crow_offsets(i); jj < src.crow_offsets(i + 1);
         jj++) {
      dst.row_indices(jj) = i;
    }
  }

  for (size_type n = 0; n < src.nnnz(); n++) {
    dst.column_indices(n) = src.ccolumn_indices(n);
  }

  for (size_type n = 0; n < src.nnnz(); n++) {
    dst.values(n) = src.cvalues(n);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_csr_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  // Complexity: Linear.  Specifically O(nnz(coo) + max(n_row,n_col))
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());
  // Make sure we zero offsets for correct cumsum
  dst.row_offsets().assign(src.nrows() + 1, 0);

  // compute number of non-zero entries per row of coo src
  for (size_type n = 0; n < src.nnnz(); n++) {
    dst.row_offsets(src.crow_indices(n))++;
  }

  // cumsum the nnz per row to get csr row_offsets
  index_type cumsum = 0;
  for (size_type i = 0; i < src.nrows(); i++) {
    index_type temp    = dst.row_offsets(i);
    dst.row_offsets(i) = cumsum;
    cumsum += temp;
  }

  dst.row_offsets(src.nrows()) = src.nnnz();

  // write coo column indices and values into csr
  for (size_type n = 0; n < src.nnnz(); n++) {
    size_type row  = src.crow_indices(n);
    size_type dest = dst.row_offsets(row);

    dst.column_indices(dest) = src.ccolumn_indices(n);
    dst.values(dest)         = src.cvalues(n);

    dst.row_offsets(row)++;
  }

  index_type last = 0;
  for (size_type i = 0; i <= src.nrows(); i++) {
    index_type temp    = dst.row_offsets(i);
    dst.row_offsets(i) = last;
    last               = temp;
  }

  // TODO: remove duplicates, if any?
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP