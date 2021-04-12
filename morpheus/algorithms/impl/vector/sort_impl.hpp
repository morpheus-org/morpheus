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

#ifndef MORPHEUS_ALGORITHMS_IMPL_VECTOR_MATRIX_SORT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_VECTOR_MATRIX_SORT_IMPL_HPP

#include <morpheus/core/type_traits.hpp>
#include <morpheus/core/exceptions.hpp>
#include <morpheus/algorithms/copy.hpp>

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

template <typename ExecSpace, typename Vector1, typename Vector2>
void counting_sort_by_key(
    const ExecSpace& space, Vector1& keys, Vector2& vals,
    typename Vector1::value_type min, typename Vector1::value_type max,
    Morpheus::DenseVectorTag,
    typename std::enable_if_t<Morpheus::is_Serial_space_v<ExecSpace>>* =
        nullptr) {
  //   using IndexType1 = typename VectorType1::value_type;
  //   using IndexType2 = typename VectorType2::value_type;

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

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_VECTOR_MATRIX_SORT_IMPL_HPP