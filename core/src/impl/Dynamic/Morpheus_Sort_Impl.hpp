/**
 * Morpheus_Sort_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_DYNAMIC_SORT_IMPL_HPP
#define MORPHEUS_DYNAMIC_SORT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Sort_Impl.hpp>
#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace>
struct sort_by_row_and_column_fn {
  using result_type = void;

  template <typename Matrix>
  result_type operator()(
      Matrix& mat, typename Matrix::index_type min_row,
      typename Matrix::index_type max_row, typename Matrix::index_type min_col,
      typename Matrix::index_type max_col,
      typename std::enable_if<
          Morpheus::is_coo_matrix_format_container_v<Matrix>>::type* =
          nullptr) {
    Impl::sort_by_row_and_column<ExecSpace>(mat, min_row, max_row, min_col,
                                            max_col);
  }

  template <typename Matrix>
  result_type operator()(
      Matrix& mat, typename Matrix::index_type, typename Matrix::index_type,
      typename Matrix::index_type, typename Matrix::index_type,
      typename std::enable_if<
          !Morpheus::is_coo_matrix_format_container_v<Matrix>>::type* =
          nullptr) {
    throw Morpheus::NotImplementedException(
        "sort_by_row_and_column() not implemented for " +
        std::to_string(mat.format_index()) + " format.");
  }
};

template <typename ExecSpace>
struct is_sorted_fn {
  using result_type = bool;

  template <typename Matrix>
  result_type operator()(
      Matrix& mat,
      typename std::enable_if<
          Morpheus::is_coo_matrix_format_container_v<Matrix>>::type* =
          nullptr) {
    return Impl::is_sorted<ExecSpace>(mat);
  }

  template <typename Matrix>
  result_type operator()(
      Matrix& mat,
      typename std::enable_if<
          !Morpheus::is_coo_matrix_format_container_v<Matrix>>::type* =
          nullptr) {
    throw Morpheus::NotImplementedException("is_sorted() not implemented for " +
                                            std::to_string(mat.format_index()) +
                                            " format.");
    return false;
  }
};

template <typename ExecSpace, typename Matrix>
void sort_by_row_and_column(
    Matrix& mat, typename Matrix::index_type min_row = 0,
    typename Matrix::index_type max_row = 0,
    typename Matrix::index_type min_col = 0,
    typename Matrix::index_type max_col = 0,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container_v<Matrix>>::type* =
        nullptr) {
  auto f = std::bind(Impl::sort_by_row_and_column_fn<ExecSpace>(),
                     std::placeholders::_1, min_row, max_row, min_col, max_col);
  Morpheus::Impl::Variant::visit(f, mat.formats());
}

template <typename ExecSpace, typename Matrix>
bool is_sorted(
    const Matrix& mat,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container_v<Matrix>>::type* =
        nullptr) {
  auto f = std::bind(Impl::is_sorted_fn<ExecSpace>(), std::placeholders::_1);
  return Morpheus::Impl::Variant::visit(f, mat.const_formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_SORT_IMPL_HPP