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

#ifndef MORPHEUS_DIA_CONVERT_IMPL_HPP
#define MORPHEUS_DIA_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <fwd/Morpheus_Fwd_Algorithms.hpp>

// TODO: Remove use of set during Coo to Dia Conversion
#include <set>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DiaTag, DiaTag,
             typename std::enable_if<
                 is_compatible_type<SourceType, DestinationType>::value ||
                 is_compatible_from_different_space<
                     SourceType, DestinationType>::value>::type* = nullptr) {
  Morpheus::copy(src, dst);
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DiaTag, CooTag) {
  using IndexType = typename SourceType::index_type;
  using ValueType = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  const IndexType ndiag = src.values.ncols();

  for (IndexType i = 0, nnzid = 0; i < ndiag; i++) {
    const IndexType k = src.diagonal_offsets[i];

    const IndexType i_start = std::max<IndexType>(0, -k);
    const IndexType j_start = std::max<IndexType>(0, k);

    // number of elements to process in this diagonal
    const IndexType N = std::min(src.nrows() - i_start, src.ncols() - j_start);

    for (IndexType n = 0; n < N; n++) {
      const ValueType temp = src.values(i_start + n, i);

      if (temp != ValueType(0)) {
        dst.row_indices[nnzid]    = i_start + n;
        dst.column_indices[nnzid] = j_start + n;
        dst.values[nnzid]         = temp;
        nnzid                     = nnzid + 1;
      }
    }
  }

  if (!dst.is_sorted()) {
    dst.sort_by_row_and_column();
  }
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, CooTag, DiaTag,
             typename SourceType::index_type alignment = 32) {
  using IndexType      = typename SourceType::index_type;
  using IndexArrayType = typename SourceType::index_array_type;

  if (src.nnnz() == 0) {
    dst.resize(src.nrows(), src.ncols(), src.nnnz(), 0);
    return;
  }

  std::vector<IndexType> diag_map(src.nnnz(), 0);

  // Find on which diagonal each entry sits on
  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    diag_map[n] = src.column_indices[n] - src.row_indices[n];
  }

  // Create unique diagonal set
  std::set<IndexType> diag_set(diag_map.begin(), diag_map.end());
  IndexType ndiags = IndexType(diag_set.size());

  // TODO: Check if fill in exceeds a tolerance value otherwise throw
  // format_conversion_exception

  IndexArrayType diagonal_offsets(ndiags, 0);
  for (auto it = diag_set.cbegin(); it != diag_set.cend(); ++it) {
    auto i              = std::distance(diag_set.cbegin(), it);
    diagonal_offsets[i] = *it;
  }

  // Create diagonal indexes
  IndexArrayType diag_idx(src.nnnz(), 0);
  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    for (IndexType i = 0; i < IndexType(ndiags); i++) {
      if (diag_map[n] == diagonal_offsets[i]) diag_idx[n] = i;
    }
  }

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), ndiags, alignment);

  for (IndexType i = 0; i < IndexType(ndiags); i++) {
    dst.diagonal_offsets[i] = diagonal_offsets[i];
  }

  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    dst.values(src.row_indices[n], diag_idx[n]) = src.values[n];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_CONVERT_IMPL_HPP