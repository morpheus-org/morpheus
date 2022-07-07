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

#ifndef MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, CsrTag, CsrTag,
    typename std::enable_if<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>::type* = nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // element-wise copy of indices and values
  for (index_type n = 0; n < src.nrows() + 1; n++) {
    dst.row_offsets(n) = src.crow_offsets(n);
  }

  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.column_indices(n) = src.ccolumn_indices(n);
  }

  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.values(n) = src.cvalues(n);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, CsrTag, CooTag,
    typename std::enable_if<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>::type* = nullptr) {
  // Complexity: Linear.  Specifically O(nnz(csr) + max(n_row,n_col))
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // expand compressed indices
  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type jj = src.crow_offsets(i); jj < src.crow_offsets(i + 1);
         jj++) {
      dst.row_indices(jj) = i;
    }
  }

  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.column_indices(n) = src.ccolumn_indices(n);
  }

  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.values(n) = src.cvalues(n);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, CooTag, CsrTag,
    typename std::enable_if<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, SourceType,
                               DestinationType>>::type* = nullptr) {
  // Complexity: Linear.  Specifically O(nnz(coo) + max(n_row,n_col))
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // compute number of non-zero entries per row of coo src
  for (index_type n = 0; n < src.nnnz(); n++) {
    dst.row_offsets(src.crow_indices(n))++;
  }

  // cumsum the nnz per row to get csr row_offsets
  for (index_type i = 0, cumsum = 0; i < src.nrows(); i++) {
    index_type temp    = dst.row_offsets(i);
    dst.row_offsets(i) = cumsum;
    cumsum += temp;
  }
  dst.row_offsets(src.nrows()) = src.nnnz();

  // write coo column indices and values into csr
  for (index_type n = 0; n < src.nnnz(); n++) {
    index_type row  = src.crow_indices(n);
    index_type dest = dst.row_offsets(row);

    dst.column_indices(dest) = src.ccolumn_indices(n);
    dst.values(dest)         = src.cvalues(n);

    dst.row_offsets(row)++;
  }

  for (index_type i = 0, last = 0; i <= src.nrows(); i++) {
    index_type temp    = dst.row_offsets(i);
    dst.row_offsets(i) = last;
    last               = temp;
  }

  // TODO: remove duplicates, if any?
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_SERIAL_CONVERT_IMPL_HPP