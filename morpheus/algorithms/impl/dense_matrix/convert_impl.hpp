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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DENSE_MATRIX_CONVERT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DENSE_MATRIX_CONVERT_IMPL_HPP

#include <morpheus/core/exceptions.hpp>
#include <morpheus/algorithms/copy.hpp>
#include <morpheus/containers/impl/format_tags.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DenseMatrixTag,
             CooTag) {
  using I = typename SourceType::index_type;
  using V = typename SourceType::value_type;

  // Count non-zeros
  I nnz = 0;
  for (I i = 0; i < src.nrows(); i++) {
    for (I j = 0; j < src.ncols(); j++) {
      if (src(i, j) != V(0)) nnz = nnz + 1;
    }
  }

  dst.resize(src.nrows(), src.ncols(), nnz);

  for (I i = 0, n = 0; i < src.nrows(); i++) {
    for (I j = 0; j < src.ncols(); j++) {
      if (src(i, j) != V(0)) {
        dst.row_indices[n]    = i;
        dst.column_indices[n] = j;
        dst.values[n]         = src(i, j);
        n                     = n + 1;
      }
    }
  }
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, CooTag,
             DenseMatrixTag) {
  using I = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols());

  for (I n = 0; n < src.nnnz(); n++) {
    I i              = dst.row_indices[n];
    I j              = dst.column_indices[n];
    dst.values(i, j) = dst.values[n];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DENSE_MATRIX_CONVERT_IMPL_HPP