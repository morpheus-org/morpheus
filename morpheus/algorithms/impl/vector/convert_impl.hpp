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

#ifndef MORPHEUS_ALGORITHMS_IMPL_VECTOR_CONVERT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_VECTOR_CONVERT_IMPL_HPP

#include <morpheus/core/exceptions.hpp>
#include <morpheus/algorithms/copy.hpp>
#include <morpheus/containers/impl/format_tags.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DenseMatrixTag,
             DenseVectorTag) {
  using I = typename SourceType::index_type;

  dst.resize(src.nrows() * src.ncols());

  for (I i = 0; i < src.nrows(); i++) {
    for (I j = 0; j < src.ncols(); j++) {
      I idx    = i * src.ncols() + j;
      dst(idx) = src(i, j);
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_VECTOR_CONVERT_IMPL_HPP