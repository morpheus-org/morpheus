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

#include <utils/Macros_Definitions.hpp>

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
 * @brief Checks the sizes of an empty CooMatrix container
 *
 */
#define CHECK_COO_EMPTY(A)                   \
  {                                          \
    EXPECT_EQ(A.nrows(), 0);                 \
    EXPECT_EQ(A.ncols(), 0);                 \
    EXPECT_EQ(A.nnnz(), 0);                  \
    EXPECT_EQ(A.row_indices().size(), 0);    \
    EXPECT_EQ(A.column_indices().size(), 0); \
    EXPECT_EQ(A.values().size(), 0);         \
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
#define VALIDATE_COO_CONTAINER(A, Aref, nnnz)                 \
  {                                                           \
    for (size_t n = 0; n < nnnz; n++) {                       \
      EXPECT_EQ(A.row_indices(n), Aref.row_indices(n));       \
      EXPECT_EQ(A.column_indices(n), Aref.column_indices(n)); \
      EXPECT_EQ(A.values(n), Aref.values(n));                 \
    }                                                         \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c.row_indices(0) = 0;  c.column_indices(0)  = 0; c.values(0)  = (value_type)1.11;
  c.row_indices(1) = 0;  c.column_indices(1)  = 3; c.values(1)  = (value_type)2.22;
  c.row_indices(2) = 0;  c.column_indices(2)  = 7; c.values(2)  = (value_type)3.33;
  c.row_indices(3) = 0;  c.column_indices(3)  = 8; c.values(3)  = (value_type)4.44;

  c.row_indices(4) = 1;  c.column_indices(4)  = 1; c.values(4)  = (value_type)5.55;
  c.row_indices(5) = 1;  c.column_indices(5)  = 4; c.values(5)  = (value_type)6.66;
  c.row_indices(6) = 1;  c.column_indices(6)  = 7; c.values(6)  = (value_type)7.77;
  c.row_indices(7) = 1;  c.column_indices(7)  = 9; c.values(7)  = (value_type)8.88;

  c.row_indices(8) = 2;  c.column_indices(8)  = 2; c.values(8)  = (value_type)9.99;
  c.row_indices(9) = 2;  c.column_indices(9)  = 5; c.values(9)  = (value_type)10.10;

  c.row_indices(10) = 3; c.column_indices(10) = 0; c.values(10) = (value_type)11.11;
  c.row_indices(11) = 3; c.column_indices(11) = 3; c.values(11) = (value_type)12.12;
  c.row_indices(12) = 3; c.column_indices(12) = 6; c.values(12) = (value_type)13.13;

  c.row_indices(13) = 4; c.column_indices(13) = 1; c.values(13) = (value_type)14.14;
  c.row_indices(14) = 4; c.column_indices(14) = 4; c.values(14) = (value_type)15.15;
  c.row_indices(15) = 4; c.column_indices(15) = 7; c.values(15) = (value_type)16.16;

  c.row_indices(16) = 5; c.column_indices(16) = 2; c.values(16) = (value_type)17.17;
  c.row_indices(17) = 5; c.column_indices(17) = 5; c.values(17) = (value_type)18.18;
  c.row_indices(18) = 5; c.column_indices(18) = 8; c.values(18) = (value_type)19.19;

  c.row_indices(19) = 6; c.column_indices(19) = 3; c.values(19) = (value_type)20.20;
  c.row_indices(20) = 6; c.column_indices(20) = 6; c.values(20) = (value_type)21.21;
  c.row_indices(21) = 6; c.column_indices(21) = 9; c.values(21) = (value_type)22.22;

  c.row_indices(22) = 7; c.column_indices(22) = 0; c.values(22) = (value_type)23.23;
  c.row_indices(23) = 7; c.column_indices(23) = 1; c.values(23) = (value_type)24.24;
  c.row_indices(24) = 7; c.column_indices(24) = 4; c.values(24) = (value_type)25.25;
  c.row_indices(25) = 7; c.column_indices(25) = 7; c.values(25) = (value_type)26.26;

  c.row_indices(26) = 8; c.column_indices(26) = 0; c.values(26) = (value_type)27.27;
  c.row_indices(27) = 8; c.column_indices(27) = 5; c.values(27) = (value_type)28.28;
  c.row_indices(28) = 8; c.column_indices(28) = 8; c.values(28) = (value_type)29.29;

  c.row_indices(29) = 9; c.column_indices(29) = 1; c.values(29) = (value_type)30.30;
  c.row_indices(30) = 9; c.column_indices(30) = 6; c.values(30) = (value_type)31.31;
  c.row_indices(31) = 9; c.column_indices(31) = 9; c.values(31) = (value_type)32.32;
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
  CHECK_COO_SIZES(c, SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ);
  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c.row_indices(0) = 0;  c.column_indices(0)  = 0; c.values(0)  = (value_type)1.11;
  c.row_indices(1) = 0;  c.column_indices(1)  = 3; c.values(1)  = (value_type)2.22;
  c.row_indices(2) = 0;  c.column_indices(2)  = 7; c.values(2)  = (value_type)3.33;
  c.row_indices(3) = 0;  c.column_indices(3)  = 8; c.values(3)  = (value_type)-4.44;

  c.row_indices(4) = 1;  c.column_indices(4)  = 1; c.values(4)  = (value_type)5.55;
  c.row_indices(5) = 1;  c.column_indices(5)  = 4; c.values(5)  = (value_type)6.66;
  c.row_indices(6) = 1;  c.column_indices(6)  = 7; c.values(6)  = (value_type)7.77;
  c.row_indices(7) = 1;  c.column_indices(7)  = 9; c.values(7)  = (value_type)-8.88;

  c.row_indices(8) = 2;  c.column_indices(8)  = 2; c.values(8)  = (value_type)9.99;
  c.row_indices(9) = 2;  c.column_indices(9)  = 5; c.values(9)  = (value_type)10.10;

  c.row_indices(10) = 3; c.column_indices(10) = 0; c.values(10) = (value_type)11.11;
  c.row_indices(11) = 3; c.column_indices(11) = 3; c.values(11) = (value_type)12.12;
  c.row_indices(12) = 3; c.column_indices(12) = 6; c.values(12) = (value_type)13.13;

  c.row_indices(13) = 4; c.column_indices(13) = 1; c.values(13) = (value_type)-14.14;
  c.row_indices(14) = 4; c.column_indices(14) = 4; c.values(14) = (value_type)-15.15;
  c.row_indices(15) = 4; c.column_indices(15) = 7; c.values(15) = (value_type)16.16;

  c.row_indices(16) = 5; c.column_indices(16) = 2; c.values(16) = (value_type)17.17;
  c.row_indices(17) = 5; c.column_indices(17) = 5; c.values(17) = (value_type)18.18;
  c.row_indices(18) = 5; c.column_indices(18) = 8; c.values(18) = (value_type)19.19;

  c.row_indices(19) = 6; c.column_indices(19) = 3; c.values(19) = (value_type)20.20;
  c.row_indices(20) = 6; c.column_indices(20) = 6; c.values(20) = (value_type)21.21;
  c.row_indices(21) = 6; c.column_indices(21) = 9; c.values(21) = (value_type)22.22;

  c.row_indices(22) = 7; c.column_indices(22) = 0; c.values(22) = (value_type)23.23;
  c.row_indices(23) = 7; c.column_indices(23) = 1; c.values(23) = (value_type)24.24;
  c.row_indices(24) = 7; c.column_indices(24) = 4; c.values(24) = (value_type)-25.25;
  c.row_indices(25) = 7; c.column_indices(25) = 7; c.values(25) = (value_type)26.26;

  c.row_indices(26) = 8; c.column_indices(26) = 0; c.values(26) = (value_type)27.27;
  c.row_indices(27) = 8; c.column_indices(27) = 5; c.values(27) = (value_type)28.28;
  c.row_indices(28) = 8; c.column_indices(28) = 8; c.values(28) = (value_type)29.29;

  c.row_indices(29) = 9; c.column_indices(29) = 1; c.values(29) = (value_type)30.30;
  c.row_indices(30) = 9; c.column_indices(30) = 6; c.values(30) = (value_type)31.31;
  c.row_indices(31) = 9; c.column_indices(31) = 9; c.values(31) = (value_type)32.32;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ);
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