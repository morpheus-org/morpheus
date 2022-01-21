/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_DIA_COPY_IMPL_HPP
#define MORPHEUS_DIA_COPY_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/DenseVector/Morpheus_Copy_Impl.hpp>
#include <impl/DenseMatrix/Morpheus_Copy_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DiaTag, DiaTag) {
  MORPHEUS_ASSERT((dst.nrows() >= src.nrows()) && (dst.ncols() >= src.ncols()),
                  "Destination matrix must have equal or larger shape to the "
                  "source matrix");
  MORPHEUS_ASSERT(dst.nnnz() >= src.nnnz(),
                  "Destination matrix must have equal or larger number of "
                  "non-zeros to the source matrix");
  MORPHEUS_ASSERT(
      dst.cdiagonal_offsets().size() >= src.cdiagonal_offsets().size(),
      "Destination matrix must have equal or larger number of diagonals to the "
      "source matrix");

  Morpheus::Impl::copy(src.cdiagonal_offsets(), dst.diagonal_offsets(),
                       DenseVectorTag(), DenseVectorTag());
  Morpheus::Impl::copy(src.cvalues(), dst.values(), DenseMatrixTag(),
                       DenseMatrixTag());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_COPY_IMPL_HPP