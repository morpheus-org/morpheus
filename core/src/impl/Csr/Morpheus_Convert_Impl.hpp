/**
 * Morpheus_Convert_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#ifndef MORPHEUS_CSR_CONVERT_IMPL_HPP
#define MORPHEUS_CSR_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Copy.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, CsrTag, CsrTag) {
  Morpheus::copy(src, dst);
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, CsrTag, CooTag) {
  // Complexity: Linear.  Specifically O(nnz(csr) + max(n_row,n_col))
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // expand compressed indices
  for (IndexType i = 0; i < src.nrows(); i++) {
    for (IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i + 1]; jj++) {
      dst.row_indices[jj] = i;
    }
  }

  Morpheus::copy(src.column_indices, dst.column_indices);
  Morpheus::copy(src.values, dst.values);
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, CooTag, CsrTag) {
  // Complexity: Linear.  Specifically O(nnz(coo) + max(n_row,n_col))
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // compute number of non-zero entries per row of coo src
  for (IndexType n = 0; n < src.nnnz(); n++) {
    dst.row_offsets[src.row_indices[n]]++;
  }

  // cumsum the nnz per row to get csr row_offsets
  for (IndexType i = 0, cumsum = 0; i < src.nrows(); i++) {
    IndexType temp     = dst.row_offsets[i];
    dst.row_offsets[i] = cumsum;
    cumsum += temp;
  }
  dst.row_offsets[src.nrows()] = src.nnnz();

  // write coo column indices and values into csr
  for (IndexType n = 0; n < src.nnnz(); n++) {
    IndexType row  = src.row_indices[n];
    IndexType dest = dst.row_offsets[row];

    dst.column_indices[dest] = src.column_indices[n];
    dst.values[dest]         = src.values[n];

    dst.row_offsets[row]++;
  }

  for (IndexType i = 0, last = 0; i <= src.nrows(); i++) {
    IndexType temp     = dst.row_offsets[i];
    dst.row_offsets[i] = last;
    last               = temp;
  }

  // TODO: remove duplicates, if any?
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_CONVERT_IMPL_HPP