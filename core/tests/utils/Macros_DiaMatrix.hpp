/**
 * Macros_DiaMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP

#include <utils/Macros_Definitions.hpp>

#include <Morpheus_Core.hpp>

/**
 * @brief Checks the sizes of a DiaMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_DIA_SIZES(A, num_rows, num_cols, num_nnz, num_diag, align) \
  {                                                                      \
    EXPECT_EQ(A.nrows(), num_rows);                                      \
    EXPECT_EQ(A.ncols(), num_cols);                                      \
    EXPECT_EQ(A.nnnz(), num_nnz);                                        \
    EXPECT_EQ(A.ndiags(), num_diag);                                     \
    EXPECT_EQ(A.alignment(), align);                                     \
    EXPECT_EQ(A.diagonal_offsets().size(), num_diag);                    \
    EXPECT_EQ(A.diagonal_offsets().view().size(), num_diag);             \
    EXPECT_EQ(A.values().nrows() * A.values().ncols(),                   \
              num_diag * align * ((num_rows + align - 1) / align));      \
    EXPECT_EQ(A.values().view().size(),                                  \
              A.values().nrows() * A.values().ncols());                  \
  }

/**
 * @brief Checks the sizes of an empty DiaMatrix container
 *
 */
#define CHECK_DIA_EMPTY(A)                     \
  {                                            \
    EXPECT_EQ(A.nrows(), 0);                   \
    EXPECT_EQ(A.ncols(), 0);                   \
    EXPECT_EQ(A.nnnz(), 0);                    \
    EXPECT_EQ(A.ndiags(), 0);                  \
    EXPECT_EQ(A.alignment(), 0);               \
    EXPECT_EQ(A.diagonal_offsets().size(), 0); \
    EXPECT_EQ(A.values().nrows(), 0);          \
    EXPECT_EQ(A.values().ncols(), 0);          \
    EXPECT_EQ(A.values().nnnz(), 0);           \
    EXPECT_EQ(A.values().view().size(), 0);    \
  }

/**
 * @brief Checks the sizes of two DiaMatrix containers if they match
 *
 */
#define CHECK_DIA_CONTAINERS(A, B)                                       \
  {                                                                      \
    EXPECT_EQ(A.nrows(), B.nrows());                                     \
    EXPECT_EQ(A.ncols(), B.ncols());                                     \
    EXPECT_EQ(A.nnnz(), B.nnnz());                                       \
    EXPECT_EQ(A.ndiags(), B.ndiags());                                   \
    EXPECT_EQ(A.alignment(), B.alignment());                             \
    EXPECT_EQ(A.diagonal_offsets().size(), B.diagonal_offsets().size()); \
    EXPECT_EQ(A.values().nrows(), B.values().nrows());                   \
    EXPECT_EQ(A.values().ncols(), B.values().ncols());                   \
    EXPECT_EQ(A.values().nnnz(), B.values().nnnz());                     \
    EXPECT_EQ(A.values().view().size(), B.values().view().size());       \
  }

/**
 * @brief Checks if the data arrays of two DiaMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_DIA_CONTAINER(A, Aref)                           \
  {                                                               \
    for (size_t n = 0; n < A.ndiags(); n++) {                     \
      EXPECT_EQ(A.diagonal_offsets(n), Aref.diagonal_offsets(n)); \
    }                                                             \
    for (size_t i = 0; i < A.values().nrows(); i++) {             \
      for (size_t j = 0; j < A.values().ncols(); j++) {           \
        EXPECT_EQ(A.values(i, j), Aref.values(i, j));             \
      }                                                           \
    }                                                             \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c.diagonal_offsets(0) = -8; 
  c.diagonal_offsets(1) = -7; 
  c.diagonal_offsets(2) = -6; 
  c.diagonal_offsets(3) = -3; 
  c.diagonal_offsets(4) = 0; 
  c.diagonal_offsets(5) = 3; 
  c.diagonal_offsets(6) = 6; 
  c.diagonal_offsets(7) = 7; 
  c.diagonal_offsets(8) = 8; 

  c.values(0, 0) = (value_type)0;     c.values(0, 1) = (value_type)0;     c.values(0, 2) = (value_type)0;
  c.values(1, 0) = (value_type)0;     c.values(1, 1) = (value_type)0;     c.values(1, 2) = (value_type)0;
  c.values(2, 0) = (value_type)0;     c.values(2, 1) = (value_type)0;     c.values(2, 2) = (value_type)0;
  c.values(3, 0) = (value_type)0;     c.values(3, 1) = (value_type)0;     c.values(3, 2) = (value_type)0;
  c.values(4, 0) = (value_type)0;     c.values(4, 1) = (value_type)0;     c.values(4, 2) = (value_type)0;
  c.values(5, 0) = (value_type)0;     c.values(5, 1) = (value_type)0;     c.values(5, 2) = (value_type)0;
  c.values(6, 0) = (value_type)0;     c.values(6, 1) = (value_type)0;     c.values(6, 2) = (value_type)0;
  c.values(7, 0) = (value_type)0;     c.values(7, 1) = (value_type)23.23; c.values(7, 2) = (value_type)24.24;
  c.values(8, 0) = (value_type)27.27; c.values(8, 1) = (value_type)0;     c.values(8, 2) = (value_type)0;
  c.values(9, 0) = (value_type)30.30; c.values(9, 1) = (value_type)0;     c.values(9, 2) = (value_type)0;

  c.values(0, 3) = (value_type)0;     c.values(0, 4) = (value_type)1.11;  c.values(0, 5) = (value_type)2.22;
  c.values(1, 3) = (value_type)0;     c.values(1, 4) = (value_type)5.55;  c.values(1, 5) = (value_type)6.66;
  c.values(2, 3) = (value_type)0;     c.values(2, 4) = (value_type)9.99;  c.values(2, 5) = (value_type)10.10;
  c.values(3, 3) = (value_type)11.11; c.values(3, 4) = (value_type)12.12; c.values(3, 5) = (value_type)13.13;
  c.values(4, 3) = (value_type)14.14; c.values(4, 4) = (value_type)15.15; c.values(4, 5) = (value_type)16.16;
  c.values(5, 3) = (value_type)17.17; c.values(5, 4) = (value_type)18.18; c.values(5, 5) = (value_type)19.19;
  c.values(6, 3) = (value_type)20.20; c.values(6, 4) = (value_type)21.21; c.values(6, 5) = (value_type)22.22;
  c.values(7, 3) = (value_type)25.25; c.values(7, 4) = (value_type)26.26; c.values(7, 5) = (value_type)0;
  c.values(8, 3) = (value_type)28.28; c.values(8, 4) = (value_type)29.29; c.values(8, 5) = (value_type)0;
  c.values(9, 3) = (value_type)31.31; c.values(9, 4) = (value_type)32.32; c.values(9, 5) = (value_type)0;

  c.values(0, 6) = (value_type)0;     c.values(0, 7) = (value_type)3.33;  c.values(0, 8) = (value_type)4.44;
  c.values(1, 6) = (value_type)7.77;  c.values(1, 7) = (value_type)0;     c.values(1, 8) = (value_type)8.88;
  c.values(2, 6) = (value_type)0;     c.values(2, 7) = (value_type)0;     c.values(2, 8) = (value_type)0;
  c.values(3, 6) = (value_type)0;     c.values(3, 7) = (value_type)0;     c.values(3, 8) = (value_type)0;
  c.values(4, 6) = (value_type)0;     c.values(4, 7) = (value_type)0;     c.values(4, 8) = (value_type)0;
  c.values(5, 6) = (value_type)0;     c.values(5, 7) = (value_type)0;     c.values(5, 8) = (value_type)0;
  c.values(6, 6) = (value_type)0;     c.values(6, 7) = (value_type)0;     c.values(6, 8) = (value_type)0;
  c.values(7, 6) = (value_type)0;     c.values(7, 7) = (value_type)0;     c.values(7, 8) = (value_type)0;
  c.values(8, 6) = (value_type)0;     c.values(8, 7) = (value_type)0;     c.values(8, 8) = (value_type)0;
  c.values(9, 6) = (value_type)0;     c.values(9, 7) = (value_type)0;     c.values(9, 8) = (value_type)0;
  // clang-format on
}

/**
 * @brief Builds a sample DiaMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A DiaMatrix type
 * @param A The DiaMatrix we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  CHECK_DIA_SIZES(c, SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
                  SMALL_DIA_MATRIX_NDIAGS, SMALL_MATRIX_ALIGNMENT);
  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // clang-format off
  c.diagonal_offsets(0) = -8; 
  c.diagonal_offsets(1) = -7; 
  c.diagonal_offsets(2) = -6; 
  c.diagonal_offsets(3) = -3; 
  c.diagonal_offsets(4) = 0; 
  c.diagonal_offsets(5) = 3; 
  c.diagonal_offsets(6) = 6; 
  c.diagonal_offsets(7) = 7; 
  c.diagonal_offsets(8) = 8; 

  c.values(0, 0) = (value_type)0;      c.values(0, 1) = (value_type)0;      c.values(0, 2) = (value_type)0;
  c.values(1, 0) = (value_type)0;      c.values(1, 1) = (value_type)0;      c.values(1, 2) = (value_type)0;
  c.values(2, 0) = (value_type)0;      c.values(2, 1) = (value_type)0;      c.values(2, 2) = (value_type)0;
  c.values(3, 0) = (value_type)0;      c.values(3, 1) = (value_type)0;      c.values(3, 2) = (value_type)0;
  c.values(4, 0) = (value_type)0;      c.values(4, 1) = (value_type)0;      c.values(4, 2) = (value_type)0;
  c.values(5, 0) = (value_type)0;      c.values(5, 1) = (value_type)0;      c.values(5, 2) = (value_type)0;
  c.values(6, 0) = (value_type)0;      c.values(6, 1) = (value_type)0;      c.values(6, 2) = (value_type)0;
  c.values(7, 0) = (value_type)0;      c.values(7, 1) = (value_type)23.23;  c.values(7, 2) = (value_type)24.24;
  c.values(8, 0) = (value_type)27.27;  c.values(8, 1) = (value_type)0;      c.values(8, 2) = (value_type)0;
  c.values(9, 0) = (value_type)30.30;  c.values(9, 1) = (value_type)0;      c.values(9, 2) = (value_type)0;
  
  c.values(0, 3) = (value_type)0;      c.values(0, 4) = (value_type)1.11;   c.values(0, 5) = (value_type)2.22;
  c.values(1, 3) = (value_type)0;      c.values(1, 4) = (value_type)5.55;   c.values(1, 5) = (value_type)6.66;
  c.values(2, 3) = (value_type)0;      c.values(2, 4) = (value_type)9.99;   c.values(2, 5) = (value_type)10.10;
  c.values(3, 3) = (value_type)11.11;  c.values(3, 4) = (value_type)12.12;  c.values(3, 5) = (value_type)13.13;
  c.values(4, 3) = (value_type)-14.14; c.values(4, 4) = (value_type)-15.15; c.values(4, 5) = (value_type)16.16;
  c.values(5, 3) = (value_type)17.17;  c.values(5, 4) = (value_type)18.18;  c.values(5, 5) = (value_type)19.19;
  c.values(6, 3) = (value_type)20.20;  c.values(6, 4) = (value_type)21.21;  c.values(6, 5) = (value_type)22.22;
  c.values(7, 3) = (value_type)-25.25; c.values(7, 4) = (value_type)26.26;  c.values(7, 5) = (value_type)0;
  c.values(8, 3) = (value_type)28.28;  c.values(8, 4) = (value_type)29.29;  c.values(8, 5) = (value_type)0;
  c.values(9, 3) = (value_type)31.31;  c.values(9, 4) = (value_type)32.32;  c.values(9, 5) = (value_type)0;
  
  c.values(0, 6) = (value_type)0;      c.values(0, 7) = (value_type)3.33;   c.values(0, 8) = (value_type)-4.44;
  c.values(1, 6) = (value_type)7.77;   c.values(1, 7) = (value_type)0;      c.values(1, 8) = (value_type)-8.88;
  c.values(2, 6) = (value_type)0;      c.values(2, 7) = (value_type)0;      c.values(2, 8) = (value_type)0;
  c.values(3, 6) = (value_type)0;      c.values(3, 7) = (value_type)0;      c.values(3, 8) = (value_type)0;
  c.values(4, 6) = (value_type)0;      c.values(4, 7) = (value_type)0;      c.values(4, 8) = (value_type)0;
  c.values(5, 6) = (value_type)0;      c.values(5, 7) = (value_type)0;      c.values(5, 8) = (value_type)0;
  c.values(6, 6) = (value_type)0;      c.values(6, 7) = (value_type)0;      c.values(6, 8) = (value_type)0;
  c.values(7, 6) = (value_type)0;      c.values(7, 7) = (value_type)0;      c.values(7, 8) = (value_type)0;
  c.values(8, 6) = (value_type)0;      c.values(8, 7) = (value_type)0;      c.values(8, 8) = (value_type)0;
  c.values(9, 6) = (value_type)0;      c.values(9, 7) = (value_type)0;      c.values(9, 8) = (value_type)0;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
           SMALL_DIA_MATRIX_NDIAGS, SMALL_MATRIX_ALIGNMENT);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container1> &&
        Morpheus::is_dia_matrix_format_container_v<Container2>>* = nullptr) {
  bool same_nrows      = c1.nrows() == c2.nrows() ? true : false;
  bool same_ncols      = c1.ncols() == c2.ncols() ? true : false;
  bool same_nnnz       = c1.nnnz() == c2.nnnz() ? true : false;
  bool same_ndiags     = c1.ndiags() == c2.ndiags() ? true : false;
  bool same_nalignment = c1.alignment() == c2.alignment() ? true : false;
  bool same_doff_size =
      c1.diagonal_offsets().size() == c2.diagonal_offsets().size() ? true
                                                                   : false;
  bool same_values_size = c1.values().nrows() * c1.values().ncols() ==
                                  c2.values().nrows() * c2.values().ncols()
                              ? true
                              : false;

  return same_nrows && same_ncols && same_nnnz && same_ndiags &&
         same_nalignment && same_doff_size && same_values_size;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  typename Container::HostMirror ch;
  ch.resize(c);
  Morpheus::copy(c, ch);

  for (size_type i = 0; i < c.values().nrows(); i++) {
    for (size_type j = 0; j < c.values().ncols(); j++) {
      if (ch.values(i, j) != (value_type)0.0) {
        return false;
      }
    }
  }

  return true;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::DiaMatrix<ValueType, IndexType, ArrayLayout, Space> create_container(
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::DiaMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container1> &&
        Morpheus::is_dia_matrix_format_container_v<Container2>>* = nullptr) {
  using size_type = typename Container1::size_type;

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  for (size_type i = 0; i < c1_h.ndiags(); i++) {
    if (c1_h.diagonal_offsets(i) != c2_h.diagonal_offsets(i)) return false;
  }

  for (size_type i = 0; i < c1_h.values().nrows(); i++) {
    for (size_type j = 0; j < c1_h.values().ncols(); j++) {
      if (c1_h.values(i, j) != c2_h.values(i, j)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP