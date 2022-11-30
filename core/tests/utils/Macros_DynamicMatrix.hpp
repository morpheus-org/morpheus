/**
 * Macros_DynamicMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DYNAMICMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_DYNAMICMATRIX_HPP

#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>

/**
 * @brief Checks the sizes of a DynamicMatrix container against a number of
 * rows, columns and non-zeros
 *
 */
#define CHECK_DYNAMIC_SIZES(A, num_rows, num_cols, num_nnz, active_index) \
  {                                                                       \
    EXPECT_EQ(A.nrows(), num_rows);                                       \
    EXPECT_EQ(A.ncols(), num_cols);                                       \
    EXPECT_EQ(A.nnnz(), num_nnz);                                         \
    EXPECT_EQ(A.formats().index(), active_index);                         \
  }

/**
 * @brief Checks the sizes of an empty DynamicMatrix container
 *
 */
#define CHECK_DYNAMIC_EMPTY(A)         \
  {                                    \
    EXPECT_EQ(A.nrows(), 0);           \
    EXPECT_EQ(A.ncols(), 0);           \
    EXPECT_EQ(A.nnnz(), 0);            \
    EXPECT_EQ(A.formats().index(), 0); \
  }

/**
 * @brief Checks the sizes of two DynamicMatrix containers if they match
 *
 */
#define CHECK_DYNAMIC_CONTAINERS(A, B)                   \
  {                                                      \
    EXPECT_EQ(A.nrows(), B.nrows());                     \
    EXPECT_EQ(A.ncols(), B.ncols());                     \
    EXPECT_EQ(A.nnnz(), B.nnnz());                       \
    EXPECT_EQ(A.formats().index(), B.formats().index()); \
  }

namespace Morpheus {
namespace Test {
namespace Impl {
struct is_same_size_fn {
  using result_type = bool;

  template <class Container1, class Container2>
  result_type operator()(
      Container1& c1, Container2& c2,
      typename std::enable_if_t<
          Morpheus::has_same_format_v<Container1, Container2>>* = nullptr) {
    return is_same_size(c1, c2);
  }

  template <class Container1, class Container2>
  result_type operator()(
      Container1&, Container2&,
      typename std::enable_if_t<
          !Morpheus::has_same_format_v<Container1, Container2>>* = nullptr) {
    return false;
  }
};

struct have_same_data_fn {
  using result_type = bool;

  template <class Container1, class Container2>
  result_type operator()(
      Container1& c1, Container2& c2,
      typename std::enable_if_t<
          Morpheus::has_same_format_v<Container1, Container2>>* = nullptr) {
    return have_same_data(c1, c2);
  }

  template <class Container1, class Container2>
  result_type operator()(
      Container1&, Container2&,
      typename std::enable_if_t<
          !Morpheus::has_same_format_v<Container1, Container2>>* = nullptr) {
    return false;
  }
};
}  // namespace Impl
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container>>* = nullptr) {
  using value_type   = typename Container::value_type;
  using index_type   = typename Container::index_type;
  using array_layout = typename Container::array_layout;
  using backend      = typename Container::backend;
  typename Morpheus::CooMatrix<value_type, index_type, array_layout,
                               backend>::HostMirror coo_mat;

  setup_small_container(coo_mat);

  c = coo_mat;
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container>>* = nullptr) {
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container1> &&
        Morpheus::is_dynamic_matrix_format_container_v<Container2>>* =
        nullptr) {
  return std::visit(Impl::is_same_size_fn(), c1.formats(), c2.formats());
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container>>* = nullptr) {
  return std::visit([&](auto&& arg) { return is_empty_container(arg); },
                    c.formats());
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::DynamicMatrix<ValueType, IndexType, ArrayLayout, Space>
create_container(
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::DynamicMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container_v<Container1> &&
        Morpheus::is_dynamic_matrix_format_container_v<Container2>>* =
        nullptr) {
  return std::visit(Impl::have_same_data_fn(), c1.formats(), c2.formats());
}
}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_DYNAMICMATRIX_HPP