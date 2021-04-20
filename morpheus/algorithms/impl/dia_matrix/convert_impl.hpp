/**
 * convert_impl.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_CONVERT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_CONVERT_IMPL_HPP

#include <set>
#include <morpheus/core/exceptions.hpp>
#include <morpheus/containers/impl/format_tags.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DiaTag, CooTag) {
  using I = typename SourceType::index_type;
  using V = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  for (I i = 0, nnzid = 0; i < I(src.diagonal_offsets.size()); i++) {
    const I k       = src.diagonal_offsets[i];  // diagonal offset
    const I i_start = std::max(0, -k);
    const I j_start = std::max(0, k);
    const I N       = std::min(src.nrows() - i_start, src.ncols() - j_start);

    for (I n = 0; n < N; n++) {
      V temp = src.values(i, j_start + n);
      if (temp != V(0)) {
        dst.row_indices[nnzid]    = i_start + n;
        dst.column_indices[nnzid] = j_start + n;
        dst.values[nnzid]         = src.values(i, j_start + n);
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
  using I                = typename SourceType::index_type;
  using index_array_type = typename SourceType::index_array_type;

  if (src.nnnz() == 0) {
    dst.resize(src.nrows(), src.ncols(), src.nnnz(), 0);
    return;
  }

  index_array_type diag_map(src.nnnz(), 0);

  // Find on which diagonal each entry sits on
  for (I n = 0; n < I(src.nnnz()); n++) {
    diag_map[n] = src.column_indices[n] - src.row_indices[n];
  }

  // Create unique diagonal set
  std::set<I> diag_set(diag_map.begin(), diag_map.end());
  I ndiags = I(diag_set.size());

  // const float max_fill   = 3.0;
  // const float threshold  = 1e6;  // 1M entries
  // const float size       = float(ndiags) * float(src.ncols());
  // const float fill_ratio = size / std::max(1.0f, float(src.nnnz()));

  // if (max_fill < fill_ratio && size > threshold)
  //   throw Morpheus::format_conversion_exception(
  //       "DiaMatrix fill-in would exceed maximum tolerance");

  index_array_type diagonal_offsets(ndiags, 0);
  for (auto it = diag_set.cbegin(); it != diag_set.cend(); ++it) {
    auto i              = std::distance(diag_set.cbegin(), it);
    diagonal_offsets[i] = *it;
  }

  // Create diagonal indexes
  index_array_type diag_idx(src.nnnz(), 0);
  for (I n = 0; n < I(src.nnnz()); n++) {
    for (I i = 0; i < I(ndiags); i++) {
      if (diag_map[n] == diagonal_offsets[i]) diag_idx[n] = i;
    }
  }

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), ndiags, alignment);

  for (I i = 0; i < I(ndiags); i++) {
    dst.diagonal_offsets[i] = diagonal_offsets[i];
  }

  for (I n = 0; n < I(src.nnnz()); n++) {
    dst.values(diag_idx[n], src.column_indices[n]) = src.values[n];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_CONVERT_IMPL_HPP