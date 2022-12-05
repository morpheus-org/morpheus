/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_DENSEMATRIX_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_DENSEMATRIX_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dense_matrix_format_container_v<SourceType> &&
        Morpheus::is_dense_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;

  MORPHEUS_ASSERT((dst.nrows() >= src.nrows()) && (dst.ncols() >= src.ncols()),
                  "Destination matrix must have equal or larger shape to the "
                  "source matrix");

  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      dst(i, j) = src(i, j);
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dense_matrix_format_container_v<SourceType> &&
        Morpheus::is_dense_vector_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows() * src.ncols());

  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      index_type idx = i * src.ncols() + j;
      dst(idx)       = src(i, j);
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dense_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;
  using value_type = typename SourceType::value_type;

  // Count non-zeros
  index_type nnz = 0;
  for (index_type i = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      if (src(i, j) != value_type(0)) nnz = nnz + 1;
    }
  }

  dst.resize(src.nrows(), src.ncols(), nnz);

  for (index_type i = 0, n = 0; i < src.nrows(); i++) {
    for (index_type j = 0; j < src.ncols(); j++) {
      if (src(i, j) != value_type(0)) {
        dst.row_indices(n)    = i;
        dst.column_indices(n) = j;
        dst.values(n)         = src(i, j);
        n                     = n + 1;
      }
    }
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_dense_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using index_type = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols());

  for (index_type n = 0; n < src.nnnz(); n++) {
    index_type i = src.crow_indices(n);
    index_type j = src.ccolumn_indices(n);
    dst(i, j)    = src.cvalues(n);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DENSEMATRIX_SERIAL_CONVERT_IMPL_HPP