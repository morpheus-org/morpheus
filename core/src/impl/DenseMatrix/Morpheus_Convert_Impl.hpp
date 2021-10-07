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

#ifndef MORPHEUS_DENSEMATRIX_CONVERT_IMPL_HPP
#define MORPHEUS_DENSEMATRIX_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <fwd/Morpheus_Fwd_Algorithms.hpp>

namespace Morpheus {
namespace Impl {

template <typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, DenseMatrixTag, DenseMatrixTag,
    typename std::enable_if<
        std::is_same<typename SourceType::memory_space,
                     typename DestinationType::memory_space>::value &&
        is_HostSpace_v<typename SourceType::memory_space>>::type* = nullptr) {
  using index_type = typename SourceType::index_type;
  dst.resize(src.nrows(), src.ncols());

  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      dst(i, j) = src(i, j);
    }
  }
}

template <typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst, DenseMatrixTag, DenseVectorTag,
    typename std::enable_if<
        std::is_same<typename SourceType::memory_space,
                     typename DestinationType::memory_space>::value &&
        is_HostSpace_v<typename SourceType::memory_space>>::type* = nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows() * src.ncols());

  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      index_type idx = i * src.ncols() + j;
      dst(idx)       = src(i, j);
    }
  }
}

template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DenseMatrixTag,
             CooTag) {
  using IndexType = typename SourceType::index_type;
  using ValueType = typename SourceType::value_type;

  // Count non-zeros
  IndexType nnz = 0;
  for (IndexType i = 0; i < src.nrows(); i++) {
    for (IndexType j = 0; j < src.ncols(); j++) {
      if (src(i, j) != ValueType(0)) nnz = nnz + 1;
    }
  }

  dst.resize(src.nrows(), src.ncols(), nnz);

  for (IndexType i = 0, n = 0; i < src.nrows(); i++) {
    for (IndexType j = 0; j < src.ncols(); j++) {
      if (src(i, j) != ValueType(0)) {
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
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols());

  for (IndexType n = 0; n < src.nnnz(); n++) {
    IndexType i = src.row_indices[n];
    IndexType j = src.column_indices[n];
    dst(i, j)   = src.values[n];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEMATRIX_CONVERT_IMPL_HPP