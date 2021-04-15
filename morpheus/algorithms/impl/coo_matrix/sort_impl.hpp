/**
 * sort_impl.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_COO_MATRIX_SORT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_COO_MATRIX_SORT_IMPL_HPP

#include <morpheus/core/exceptions.hpp>
#include <morpheus/core/type_traits.hpp>
#include <morpheus/containers/impl/format_tags.hpp>
#include <morpheus/algorithms/impl/vector/sort_impl.hpp>

namespace Morpheus {
namespace Impl {
template <typename ExecSpace, typename Matrix>
void sort_by_row_and_column(
    const ExecSpace& space, Matrix& mat, CooTag,
    typename Matrix::index_type min_row = 0,
    typename Matrix::index_type max_row = 0,
    typename Matrix::index_type min_col = 0,
    typename Matrix::index_type max_col = 0,
    typename std::enable_if_t<Morpheus::is_Serial_space_v<ExecSpace>>* =
        nullptr) {
  using IndexType      = typename Matrix::index_type;
  using IndexArrayType = typename Matrix::index_array_type;
  using ValueArrayType = typename Matrix::value_array_type;

  IndexType N = mat.row_indices.size();

  IndexArrayType permutation("permutation", N);
  for (IndexType i = 0; i < N; i++) permutation[i] = i;

  IndexType minr = min_row;
  IndexType maxr = max_row;
  IndexType minc = min_col;
  IndexType maxc = max_col;

  if (maxr == 0) {
    maxr = Morpheus::Impl::find_max_element(mat.row_indices);
  }
  if (maxc == 0) {
    maxc = Morpheus::Impl::find_max_element(mat.column_indices);
  }

  {
    IndexArrayType temp;
    Morpheus::copy(mat.column_indices, temp);

    Morpheus::Impl::counting_sort_by_key(space, temp, permutation, minc, maxc,
                                         typename IndexArrayType::tag());

    Morpheus::copy(mat.row_indices, temp);

    for (IndexType i = 0; i < IndexType(permutation.size()); i++) {
      mat.row_indices[i] = temp[permutation[i]];
    }

    Morpheus::Impl::counting_sort_by_key(space, mat.row_indices, permutation,
                                         minr, maxr,
                                         typename IndexArrayType::tag());
    Morpheus::copy(mat.column_indices, temp);
    for (IndexType i = 0; i < IndexType(permutation.size()); i++) {
      mat.column_indices[i] = temp[permutation[i]];
    }
  }

  {
    ValueArrayType temp;
    Morpheus::copy(mat.values, temp);
    for (IndexType i = 0; i < IndexType(permutation.size()); i++) {
      mat.values[i] = temp[permutation[i]];
    }
  }
}

template <typename ExecSpace, typename Matrix>
bool is_sorted(
    const ExecSpace& space, Matrix& mat, CooTag,
    typename std::enable_if_t<Morpheus::is_Serial_space_v<ExecSpace>>* =
        nullptr) {
  using IndexType = typename Matrix::index_array_type::index_type;

  if (mat.row_indices.size() != mat.column_indices.size()) {
    throw Morpheus::RuntimeException(
        "Sizes of row and column indeces do not match.");
  }

  for (IndexType i = 0; i < IndexType(mat.nnnz()) - 1; i++) {
    if ((mat.row_indices[i] > mat.row_indices[i + 1]) ||
        (mat.row_indices[i] == mat.row_indices[i + 1] &&
         mat.column_indices[i] > mat.column_indices[i + 1]))
      return false;
  }
  return true;
}
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_COO_MATRIX_SORT_IMPL_HPP