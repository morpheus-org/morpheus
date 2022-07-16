/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_CSR_COPY_IMPL_HPP
#define MORPHEUS_CSR_COPY_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/DenseVector/Morpheus_Copy_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_csr_matrix_format_container_v<SourceType> &&
        Morpheus::is_csr_matrix_format_container_v<DestinationType>>::type* =
        nullptr) {
  MORPHEUS_ASSERT(
      (dst.nrows() == src.nrows()) && (dst.ncols() == src.ncols()),
      "Destination matrix must have equal shape to the source matrix");
  MORPHEUS_ASSERT(dst.nnnz() == src.nnnz(),
                  "Destination matrix must have equal number of non-zeros to "
                  "the source matrix");

  Morpheus::Impl::copy(src.crow_offsets(), dst.row_offsets(), DenseVectorTag(),
                       DenseVectorTag());
  Morpheus::Impl::copy(src.ccolumn_indices(), dst.column_indices(),
                       DenseVectorTag(), DenseVectorTag());
  Morpheus::Impl::copy(src.cvalues(), dst.values(), DenseVectorTag(),
                       DenseVectorTag());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_COPY_IMPL_HPP