/**
 * Macros_CooMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_COOMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_COOMATRIX_HPP

#include <Morpheus_Core.hpp>

/**
 * @brief Checks the sizes of a CooMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_COO_SIZES(A, num_rows, num_cols, num_nnz) \
  {                                                     \
    EXPECT_EQ(A.nrows(), num_rows);                     \
    EXPECT_EQ(A.ncols(), num_cols);                     \
    EXPECT_EQ(A.nnnz(), num_nnz);                       \
    EXPECT_EQ(A.row_indices().size(), num_nnz);         \
    EXPECT_EQ(A.column_indices().size(), num_nnz);      \
    EXPECT_EQ(A.values().size(), num_nnz);              \
  }

/**
 * @brief Checks the sizes of two CooMatrix containers if they match
 *
 */
#define CHECK_COO_CONTAINERS(A, B)                                   \
  {                                                                  \
    EXPECT_EQ(A.nrows(), B.nrows());                                 \
    EXPECT_EQ(A.ncols(), B.ncols());                                 \
    EXPECT_EQ(A.nnnz(), B.nnnz());                                   \
    EXPECT_EQ(A.row_indices().size(), B.row_indices().size());       \
    EXPECT_EQ(A.column_indices().size(), B.column_indices().size()); \
    EXPECT_EQ(A.values().size(), B.values().size());                 \
  }

/**
 * @brief Checks if the data arrays of two CooMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_COO_CONTAINER(A, Aref, nnnz)                       \
  {                                                                 \
    using container_type      = decltype(A);                        \
    using container_size_type = typename container_type::size_type; \
    for (container_size_type n = 0; n < nnnz; n++) {                \
      EXPECT_EQ(A.row_indices(n), Aref.row_indices(n));             \
      EXPECT_EQ(A.column_indices(n), Aref.column_indices(n));       \
      EXPECT_EQ(A.values(n), Aref.values(n));                       \
    }                                                               \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // Matrix
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]

  // clang-format off
  c.row_indices(0) = 0; c.column_indices(0) = 0; c.values(0) = (value_type)1.11;
  c.row_indices(1) = 0; c.column_indices(1) = 2; c.values(1) = (value_type)2.22;
  c.row_indices(2) = 1; c.column_indices(2) = 2; c.values(2) = (value_type)3.33;
  c.row_indices(3) = 2; c.column_indices(3) = 1; c.values(3) = (value_type)4.44;
  // clang-format on
}

/**
 * @brief Builds a sample CooMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A CooMatrix type
 * @param A The CooMatrix we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_COO_SIZES(c, 3, 3, 4);

  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // New Matrix
  // [1.11 *    *    ]
  // [*    *    -3.33]
  // [2.22 4.44 *    ]

  // clang-format off
  c.row_indices(0) = 0; c.column_indices(0) = 0; c.values(0) = (value_type)1.11;
  c.row_indices(1) = 1; c.column_indices(1) = 2; c.values(1) = (value_type)-3.33;
  c.row_indices(2) = 2; c.column_indices(2) = 0; c.values(2) = (value_type)2.22;
  c.row_indices(3) = 2; c.column_indices(3) = 1; c.values(3) = (value_type)4.44;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(3, 3, 4);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container1> &&
        Morpheus::is_coo_matrix_format_container_v<Container2>>* = nullptr) {
  bool same_nrows = c1.nrows() == c2.nrows() ? true : false;
  bool same_ncols = c1.ncols() == c2.ncols() ? true : false;
  bool same_nnnz  = c1.nnnz() == c2.nnnz() ? true : false;
  bool same_rind_size =
      c1.row_indices().size() == c2.row_indices().size() ? true : false;
  bool same_cind_size =
      c1.column_indices().size() == c2.column_indices().size() ? true : false;
  bool same_values_size =
      c1.values().size() == c2.values().size() ? true : false;

  return same_nrows && same_ncols && same_nnnz && same_rind_size &&
         same_cind_size && same_values_size;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  typename Container::HostMirror ch;
  ch.resize(c);
  Morpheus::copy(c, ch);

  for (size_type n = 0; n < c.nnnz(); n++) {
    if (ch.values(n) != (value_type)0.0) {
      return false;
    }
  }

  return true;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout, Space> create_container(
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container1> &&
        Morpheus::is_coo_matrix_format_container_v<Container2>>* = nullptr) {
  using size_type = typename Container1::size_type;

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  for (size_type i = 0; i < c1_h.nnnz(); i++) {
    if (c1_h.row_indices(i) != c2_h.row_indices(i)) return false;
    if (c1_h.column_indices(i) != c2_h.column_indices(i)) return false;
    if (c1_h.values(i) != c2_h.values(i)) return false;
  }

  return true;
}

}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_COOMATRIX_HPP