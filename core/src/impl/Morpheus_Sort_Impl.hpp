/**
 * Morpheus_Sort_Impl.hpp
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

#ifndef MORPHEUS_SORT_IMPL_HPP
#define MORPHEUS_SORT_IMPL_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <fwd/Morpheus_Fwd_Algorithms.hpp>

namespace Morpheus {
namespace Impl {

template <typename Vector>
typename Vector::value_type find_max_element(const Vector& vec) {
  using IndexType = typename Vector::index_type;
  IndexType max   = 0;

  for (IndexType i = 0; i < vec.size(); i++) {
    if (size_t(vec[i]) > max) max = size_t(vec[i]);
  }

  return max;
}

template <typename Vector1, typename Vector2>
void counting_sort_by_key(Vector1& keys, Vector2& vals,
                          typename Vector1::value_type min,
                          typename Vector1::value_type max, DenseVectorTag) {
  if (min < 0)
    throw Morpheus::InvalidInputException(
        "counting_sort_by_key min element less than 0");

  if (max < min)
    throw Morpheus::InvalidInputException(
        "counting_sort_by_key min element less than max element");

  if (keys.size() < vals.size())
    throw Morpheus::InvalidInputException(
        "counting_sort_by_key keys.size() less than vals.size()");

  if (min > 0) min = 0;

  // compute the number of bins
  size_t size = max - min;

  // allocate temporary arrays
  Vector1 counts(size + 2, 0), temp_keys, temp_vals;
  Morpheus::copy(keys, temp_keys);
  Morpheus::copy(vals, temp_vals);

  // count the number of occurences of each key
  for (size_t i = 0; i < keys.size(); i++) counts[keys[i] + 1]++;

  // scan the sum of each bin
  for (size_t i = 0; i < size; i++) counts[i + 1] += counts[i];

  // generate output in sorted order
  for (size_t i = 0; i < keys.size(); i++) {
    keys[counts[temp_keys[i]]]   = temp_keys[i];
    vals[counts[temp_keys[i]]++] = temp_vals[i];
  }
}

template <typename Matrix>
void sort_by_row_and_column(Matrix& mat, CooTag,
                            typename Matrix::index_type min_row = 0,
                            typename Matrix::index_type max_row = 0,
                            typename Matrix::index_type min_col = 0,
                            typename Matrix::index_type max_col = 0) {
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

    Morpheus::Impl::counting_sort_by_key(temp, permutation, minc, maxc,
                                         typename IndexArrayType::tag());

    Morpheus::copy(mat.row_indices, temp);

    for (IndexType i = 0; i < IndexType(permutation.size()); i++) {
      mat.row_indices[i] = temp[permutation[i]];
    }

    Morpheus::Impl::counting_sort_by_key(mat.row_indices, permutation, minr,
                                         maxr, typename IndexArrayType::tag());
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

template <typename Matrix>
bool is_sorted(Matrix& mat, CooTag) {
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

#endif  // MORPHEUS_SORT_IMPL_HPP