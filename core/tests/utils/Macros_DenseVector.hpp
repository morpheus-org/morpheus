/**
 * Macros_DenseVector.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DENSEVECTOR_HPP
#define TEST_CORE_UTILS_MACROS_DENSEVECTOR_HPP

#include <Morpheus_Core.hpp>

/**
 * @brief Checks the sizes of a DenseVector container against a size
 *
 */
#define CHECK_DENSE_VECTOR_SIZES(v, vec_size) \
  {                                           \
    EXPECT_EQ(v.size(), vec_size);            \
    EXPECT_EQ(v.view().size(), vec_size);     \
  }

/**
 * @brief Checks the sizes of two DenseVector containers if they match
 *
 */
#define CHECK_DENSE_VECTOR_CONTAINERS(v1, v2)      \
  {                                                \
    EXPECT_EQ(v1.size(), v2.size());               \
    EXPECT_EQ(v1.view().size(), v2.view().size()); \
  }

/**
 * @brief Checks if the data arrays of two DenseVector containers contain the
 * same data.
 *
 */
#define VALIDATE_DENSE_VECTOR_CONTAINER(v1, v2, size, type) \
  {                                                         \
    for (type n = 0; n < size; n++) {                       \
      EXPECT_EQ(v1[n], v2[n]);                              \
    }                                                       \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  using value_type = typename Container::value_type;
  using index_type = typename Container::index_type;

  for (index_type i = 0; i < (index_type)c.size(); i++) {
    c[i] = (value_type)1.11 * (value_type)i;
  }
}

/**
 * @brief Builds a sample DenseVector container. Assumes we have already
 * constructed the vector and we are only adding data.
 *
 * @tparam Vector A DenseVector type
 * @param v The DenseVector we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  CHECK_DENSE_VECTOR_SIZES(c, 3);

  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  using value_type = typename Container::value_type;
  using value_type = typename Container::value_type;
  using index_type = typename Container::index_type;

  for (index_type i = 0; i < (index_type)c.size(); i++) {
    c[i] = (value_type)-1.11 * (value_type)i;
  }
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  c.resize(3);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container1> &&
                              Morpheus::is_vector_container_v<Container2>>* =
        nullptr) {
  bool same_size        = c1.size() == c2.size() ? true : false;
  bool same_values_size = c1.view().size() == c2.view().size() ? true : false;

  return same_size && same_values_size;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  using value_type = typename Container::value_type;
  using index_type = typename Container::index_type;

  typename Container::HostMirror ch;
  ch.resize(c);
  Morpheus::copy(c, ch);

  for (index_type i = 0; i < (index_type)c.size(); i++) {
    if (ch[i] != (value_type)0.0) {
      return false;
    }
  }

  return true;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::DenseVector<ValueType, IndexType, ArrayLayout, Space>
create_container(
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container>>* =
        nullptr) {
  return Morpheus::DenseVector<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container1> &&
                              Morpheus::is_vector_container_v<Container2>>* =
        nullptr) {
  using index_type = typename Container1::index_type;

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  for (index_type i = 0; i < (index_type)c1_h.size(); i++) {
    if (c1_h[i] != c2_h[i]) {
      return false;
    }
  }

  return true;
}

template <class Container1, class Container2>
bool have_approx_same_data(
    Container1& c1, Container2& c2, bool verbose = false,
    typename std::enable_if_t<Morpheus::is_vector_container_v<Container1> &&
                              Morpheus::is_vector_container_v<Container2>>* =
        nullptr) {
  using value_type = typename Container1::value_type;
  double epsilon   = 1.0e-14;
  if (std::is_same_v<value_type, double>) {
    epsilon = 1.0e-13;
  } else if (std::is_same_v<value_type, float>) {
    epsilon = 1.0e-5;
  } else {
    epsilon = 0;
  }

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  bool res = true;
  for (size_t i = 0; i < c1_h.size(); i++) {
    if (fabs(c1_h[i] - c2_h[i]) > epsilon) {
      if (verbose) {
        std::cout << "Entry at " << i << " differs! fabs(" << c1_h[i] << " - "
                  << c2_h[i] << ") = " << fabs(c1_h[i] - c2_h[i]) << std::endl;
      }
      res = false;
    }
  }
  return res;
}  // namespace Test
}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_DENSEVECTOR_HPP