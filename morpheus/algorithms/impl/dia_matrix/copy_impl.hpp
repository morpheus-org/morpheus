/**
 * copy_impl.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_COPY_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_COPY_IMPL_HPP

#include <morpheus/containers/dia_matrix.hpp>
#include <morpheus/algorithms/impl/dense_matrix/copy_impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, Morpheus::DiaTag,
          Morpheus::DiaTag) {
  dst.resize(src.nrows(), src.ncols(), src.nnnz(), src.diagonal_offsets.size());

  Morpheus::copy(src.diagonal_offsets, dst.diagonal_offsets);
  Morpheus::copy(src.values, dst.values);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_COPY_IMPL_HPP