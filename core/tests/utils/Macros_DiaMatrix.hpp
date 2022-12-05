/**
 * Macros_DiaMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP

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
#define VALIDATE_DIA_CONTAINER(A, Aref, type)                     \
  {                                                               \
    for (type n = 0; n < A.ndiags(); n++) {                       \
      EXPECT_EQ(A.diagonal_offsets(n), Aref.diagonal_offsets(n)); \
    }                                                             \
    for (type i = 0; i < A.values().nrows(); i++) {               \
      for (type j = 0; j < A.values().ncols(); j++) {             \
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
  // Matrix
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]

  // clang-format off
  c.diagonal_offsets(0) = -1; 
  c.diagonal_offsets(1) = 0; 
  c.diagonal_offsets(2) = 1; 
  c.diagonal_offsets(3) = 2; 
  // values are:
  // [*    1.11 0    2.22]
  // [0    0    3.33 *]
  // [4.44 0    *    *]
  c.values(0,0) = (value_type)0;    c.values(1,0) = (value_type)0;    c.values(2,0) = (value_type)4.44;
  c.values(0,1) = (value_type)1.11; c.values(1,1) = (value_type)0;    c.values(2,1) = (value_type)0;
  c.values(0,2) = (value_type)0;    c.values(1,2) = (value_type)3.33; c.values(2,2) = (value_type)0;
  c.values(0,3) = (value_type)2.22; c.values(1,3) = (value_type)0;    c.values(2,3) = (value_type)0;
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
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_DIA_SIZES(c, 3, 3, 4, 4, 32);

  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;
  // New Matrix
  // [1.11 *    *    ]
  // [*    *    -3.33]
  // [2.22 4.44 *    ]

  // clang-format off
  c.diagonal_offsets(0) = -2; 
  c.diagonal_offsets(1) = -1; 
  c.diagonal_offsets(2) = 0; 
  c.diagonal_offsets(3) = 1; 
  // values are:
  // [*    *    1.11 0    ]
  // [*    0    0    -3.33]
  // [2.22 4.44 0    *    ]

  c.values(0,0) = (value_type)0;    c.values(1,0) = (value_type)0;    c.values(2,0) = (value_type)2.22;
  c.values(0,1) = (value_type)0;    c.values(1,1) = (value_type)0;    c.values(2,1) = (value_type)4.44;
  c.values(0,2) = (value_type)1.11; c.values(1,2) = (value_type)0;    c.values(2,2) = (value_type)0;
  c.values(0,3) = (value_type)0;    c.values(1,3) = (value_type)3.33; c.values(2,3) = (value_type)0;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(3, 3, 4, 4, 32);
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
  using index_type = typename Container::index_type;

  typename Container::HostMirror ch;
  ch.resize(c);
  Morpheus::copy(c, ch);

  for (index_type i = 0; i < c.values().nrows(); i++) {
    for (index_type j = 0; j < c.values().ncols(); j++) {
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
  using index_type = typename Container1::index_type;

  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  for (index_type i = 0; i < c1_h.ndiags(); i++) {
    if (c1_h.diagonal_offsets(i) != c2_h.diagonal_offsets(i)) return false;
  }

  for (index_type i = 0; i < c1_h.values().nrows(); i++) {
    for (index_type j = 0; j < c1_h.values().ncols(); j++) {
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