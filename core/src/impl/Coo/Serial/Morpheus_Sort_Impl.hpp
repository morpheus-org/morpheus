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

#ifndef MORPHEUS_COO_SERIAL_SORT_IMPL_HPP
#define MORPHEUS_COO_SERIAL_SORT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/DenseVector/Morpheus_Copy_Impl.hpp>

#include <limits>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type find_max_element(
    const Vector& vec,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using value_type = typename Vector::value_type;
  using size_type  = typename Vector::size_type;

  value_type max = std::numeric_limits<value_type>::min();

  for (size_type i = 0; i < vec.size(); i++) {
    if (vec[i] > max) max = vec[i];
  }

  return max;
}

template <typename ExecSpace, typename Vector1, typename Vector2>
void counting_sort_by_key(
    Vector1& keys, Vector2& vals, typename Vector1::value_type min,
    typename Vector1::value_type max,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using size_type1 = typename Vector1::size_type;
  using size_type2 = typename Vector2::size_type;
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
  size_type1 size = max - min;

  // allocate temporary arrays
  Vector1 counts(size + 2, 0), temp_keys(keys.size());
  Vector2 temp_vals(vals.size());
  Impl::copy(keys, temp_keys);
  Impl::copy(vals, temp_vals);

  // count the number of occurences of each key
  for (size_type1 i = 0; i < keys.size(); i++) counts[keys[i] + 1]++;

  // scan the sum of each bin
  for (size_type1 i = 0; i < size; i++) counts[i + 1] += counts[i];

  // generate output in sorted order
  for (size_type2 i = 0; i < keys.size(); i++) {
    keys[counts[temp_keys[i]]]   = temp_keys[i];
    vals[counts[temp_keys[i]]++] = temp_vals[i];
  }
}

template <typename ExecSpace, typename Matrix>
void sort_by_row_and_column(
    Matrix& mat, typename Matrix::index_type min_row = 0,
    typename Matrix::index_type max_row = 0,
    typename Matrix::index_type min_col = 0,
    typename Matrix::index_type max_col = 0,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  using index_type  = typename Matrix::index_type;
  using size_type   = typename Matrix::size_type;
  using index_array = typename Matrix::index_array_type;
  using value_array = typename Matrix::value_array_type;

  size_type N = mat.row_indices().size();

  index_array permutation(N);
  for (size_type i = 0; i < N; i++) permutation[i] = i;

  index_type minr = min_row;
  index_type maxr = max_row;
  index_type minc = min_col;
  index_type maxc = max_col;

  if (maxr == 0) {
    maxr = Impl::find_max_element<ExecSpace>(mat.row_indices());
  }
  if (maxc == 0) {
    maxc = Impl::find_max_element<ExecSpace>(mat.column_indices());
  }

  {
    index_array temp(mat.column_indices().size());
    Impl::copy(mat.column_indices(), temp);
    Impl::counting_sort_by_key<ExecSpace>(temp, permutation, minc, maxc);

    if (mat.row_indices().size() != temp.size()) {
      temp.resize(mat.row_indices().size());
    }
    Impl::copy(mat.row_indices(), temp);
    Impl::copy_by_key<ExecSpace>(permutation, temp, mat.row_indices());

    Impl::counting_sort_by_key<ExecSpace>(mat.row_indices(), permutation, minr,
                                          maxr);
    Impl::copy(mat.column_indices(), temp);
    Impl::copy_by_key<ExecSpace>(permutation, temp, mat.column_indices());
  }

  {
    value_array temp(mat.values().size());
    Impl::copy(mat.values(), temp);
    Impl::copy_by_key<ExecSpace>(permutation, temp, mat.values());
  }
}

template <typename ExecSpace, typename Matrix>
bool is_sorted(Matrix& mat,
               typename std::enable_if_t<
                   Morpheus::is_coo_matrix_format_container_v<Matrix> &&
                   Morpheus::has_custom_backend_v<ExecSpace> &&
                   Morpheus::has_serial_execution_space_v<ExecSpace> &&
                   Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  using size_type = typename Matrix::size_type;

  if (mat.row_indices().size() != mat.column_indices().size()) {
    throw Morpheus::RuntimeException(
        "Sizes of row and column indeces do not match.");
  }

  for (size_type i = 0; i < mat.nnnz() - 1; i++) {
    if ((mat.row_indices(i) > mat.row_indices(i + 1)) ||
        (mat.row_indices(i) == mat.row_indices(i + 1) &&
         mat.column_indices(i) > mat.column_indices(i + 1)))
      return false;
  }
  return true;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_COO_SERIAL_SORT_IMPL_HPP