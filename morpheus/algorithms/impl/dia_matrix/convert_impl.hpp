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
void convert(const SourceType& src, DestinationType& dst, CooTag, DiaTag) {
  throw Morpheus::NotImplementedException(
      "convert(const SourceType& src, DestinationType& "
      "dst, CooTag, DiaTag)");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_CONVERT_IMPL_HPP