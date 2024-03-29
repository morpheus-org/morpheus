/**
 * Macros_DenseMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DENSEMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_DENSEMATRIX_HPP

#include <utils/Macros_Definitions.hpp>

#include <Morpheus_Core.hpp>

/**
 * @brief Checks the sizes of a DenseMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_DENSE_MATRIX_SIZES(A, num_rows, num_cols, num_nnz) \
  {                                                              \
    EXPECT_EQ(A.nrows(), num_rows);                              \
    EXPECT_EQ(A.ncols(), num_cols);                              \
    EXPECT_EQ(A.nnnz(), num_nnz);                                \
    EXPECT_EQ(A.view().size(), num_rows* num_cols);              \
  }

/**
 * @brief Checks the sizes of two DenseMatrix containers if they match
 *
 */
#define CHECK_DENSE_MATRIX_CONTAINERS(A, B)      \
  {                                              \
    EXPECT_EQ(A.nrows(), B.nrows());             \
    EXPECT_EQ(A.ncols(), B.ncols());             \
    EXPECT_EQ(A.nnnz(), B.nnnz());               \
    EXPECT_EQ(A.view().size(), B.view().size()); \
  }

/**
 * @brief Checks if the data arrays of two DenseMatrix containers contain the
 * same data.
 *
 */
#define VALIDATE_DENSE_MATRIX_CONTAINER(A, Aref, nrows, ncols) \
  {                                                            \
    for (size_t i = 0; i < nrows; i++) {                       \
      for (type j = 0; j < ncols; j++) {                       \
        EXPECT_EQ(A(i, j), Aref(i, j));                        \
      }                                                        \
    }                                                          \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c(0, 0) = (value_type)1.11;
  c(0, 3) = (value_type)2.22;
  c(0, 7) = (value_type)3.33;
  c(0, 8) = (value_type)4.44;
  c(1, 1) = (value_type)5.55;
  c(1, 4) = (value_type)6.66;
  c(1, 7) = (value_type)7.77;
  c(1, 9) = (value_type)8.88;
  c(2, 2) = (value_type)9.99;
  c(2, 5) = (value_type)10.10;
  c(3, 0) = (value_type)11.11;
  c(3, 3) = (value_type)12.12;
  c(3, 6) = (value_type)13.13;
  c(4, 1) = (value_type)14.14;
  c(4, 4) = (value_type)15.15;
  c(4, 7) = (value_type)16.16;
  c(5, 2) = (value_type)17.17;
  c(5, 5) = (value_type)18.18;
  c(5, 8) = (value_type)19.19;
  c(6, 3) = (value_type)20.20;
  c(6, 6) = (value_type)21.21;
  c(6, 9) = (value_type)22.22;
  c(7, 0) = (value_type)23.23;
  c(7, 1) = (value_type)24.24;
  c(7, 4) = (value_type)25.25;
  c(7, 7) = (value_type)26.26;
  c(8, 0) = (value_type)27.27;
  c(8, 5) = (value_type)28.28;
  c(8, 8) = (value_type)29.29;
  c(9, 1) = (value_type)30.30;
  c(9, 6) = (value_type)31.31;
  c(9, 9) = (value_type)32.32;
  // clang-format on
}

/**
 * @brief Builds a sample DenseMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A DenseMatrix type
 * @param A The DenseMatrix we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  CHECK_DENSE_MATRIX_SIZES(c, SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS,
                           SMALL_MATRIX_NROWS * SMALL_MATRIX_NCOLS);
  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c(0, 0) = (value_type)1.11;
  c(0, 3) = (value_type)2.22;
  c(0, 7) = (value_type)3.33;
  c(0, 8) = (value_type)-4.44;
  c(1, 1) = (value_type)5.55;
  c(1, 4) = (value_type)6.66;
  c(1, 7) = (value_type)7.77;
  c(1, 9) = (value_type)-8.88;
  c(2, 2) = (value_type)9.99;
  c(2, 5) = (value_type)10.10;
  c(3, 0) = (value_type)11.11;
  c(3, 3) = (value_type)12.12;
  c(3, 6) = (value_type)13.13;
  c(4, 1) = (value_type)-14.14;
  c(4, 4) = (value_type)-15.15;
  c(4, 7) = (value_type)16.16;
  c(5, 2) = (value_type)17.17;
  c(5, 5) = (value_type)18.18;
  c(5, 8) = (value_type)19.19;
  c(6, 3) = (value_type)20.20;
  c(6, 6) = (value_type)21.21;
  c(6, 9) = (value_type)22.22;
  c(7, 0) = (value_type)23.23;
  c(7, 1) = (value_type)24.24;
  c(7, 4) = (value_type)-25.25;
  c(7, 7) = (value_type)26.26;
  c(8, 0) = (value_type)27.27;
  c(8, 5) = (value_type)28.28;
  c(8, 8) = (value_type)29.29;
  c(9, 1) = (value_type)30.30;
  c(9, 6) = (value_type)31.31;
  c(9, 9) = (value_type)32.32;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container1> &&
        Morpheus::is_dense_matrix_format_container_v<Container2>>* = nullptr) {
  bool same_nrows       = c1.nrows() == c2.nrows() ? true : false;
  bool same_ncols       = c1.ncols() == c2.ncols() ? true : false;
  bool same_nnnz        = c1.nnnz() == c2.nnnz() ? true : false;
  bool same_values_size = c1.view().size() == c2.view().size() ? true : false;

  return same_nrows && same_ncols && same_nnnz && same_values_size;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  typename Container::HostMirror ch;
  ch.resize(c);
  Morpheus::copy(c, ch);

  for (size_type i = 0; i < c.nrows(); i++) {
    for (size_type j = 0; j < c.ncols(); j++) {
      if (ch(i, j) != (value_type)0.0) {
        return false;
      }
    }
  }

  return true;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout, Space>
create_container(
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dense_matrix_format_container_v<Container1> &&
        Morpheus::is_dense_matrix_format_container_v<Container2>>* = nullptr) {
  using size_type = typename Container1::size_type;

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  for (size_type i = 0; i < c1_h.nrows(); i++) {
    for (size_type j = 0; j < c1_h.ncols(); j++) {
      if (c1_h(i, j) != c2_h(i, j)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_COOMATRIX_HPP