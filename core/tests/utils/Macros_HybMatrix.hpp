/**
 * Macros_HybMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_HYBMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_HYBMATRIX_HPP

#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_EllMatrix.hpp>

#include <Morpheus_Core.hpp>

/**
 * @brief Checks the sizes of a HybMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_HYB_SIZES(A, num_rows, num_cols, num_ell_entries,      \
                        num_coo_entries, num_entries_per_row, align) \
  {                                                                  \
    EXPECT_EQ(A.nrows(), num_rows);                                  \
    EXPECT_EQ(A.ncols(), num_cols);                                  \
    EXPECT_EQ(A.nnnz(), num_ell_entries + num_coo_entries);          \
    EXPECT_EQ(A.alignment(), align);                                 \
    CHECK_ELL_SIZES(A.ell(), num_rows, num_cols, num_ell_entries,    \
                    num_entries_per_row, align);                     \
    CHECK_COO_SIZES(A.coo(), num_rows, num_cols, num_coo_entries);   \
  }

/**
 * @brief Checks the sizes of an empty HybMatrix container
 *
 */
#define CHECK_HYB_EMPTY(A)       \
  {                              \
    EXPECT_EQ(A.nrows(), 0);     \
    EXPECT_EQ(A.ncols(), 0);     \
    EXPECT_EQ(A.nnnz(), 0);      \
    EXPECT_EQ(A.alignment(), 0); \
    CHECK_ELL_EMPTY(A.ell());    \
    CHECK_COO_EMPTY(A.coo());    \
  }

/**
 * @brief Checks the sizes of two HybMatrix containers if they match
 *
 */
#define CHECK_HYB_CONTAINERS(A, B)           \
  {                                          \
    EXPECT_EQ(A.nrows(), B.nrows());         \
    EXPECT_EQ(A.ncols(), B.ncols());         \
    EXPECT_EQ(A.nnnz(), B.nnnz());           \
    EXPECT_EQ(A.alignment(), B.alignment()); \
    CHECK_ELL_CONTAINERS(A.ell(), B.ell());  \
    CHECK_COO_CONTAINERS(A.coo(), B.coo());  \
  }

/**
 * @brief Checks if the data arrays of two HybMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_HYB_CONTAINER(A, Aref)                          \
  {                                                              \
    VALIDATE_ELL_CONTAINER(A.ell(), Aref.ell());                 \
    VALIDATE_COO_CONTAINER(A.coo(), Aref.coo(), A.coo().nnnz()); \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  using size_type  = typename Container::size_type;
  using value_type = typename Container::value_type;

  for (size_type i = 0; i < c.ell().values().nrows(); i++) {
    for (size_type j = 0; j < c.ell().values().ncols(); j++) {
      c.ell().column_indices(i, j) = c.ell().invalid_index();
      c.ell().values(i, j)         = value_type(0);
    }
  }

  for (size_type n = 0; n < c.coo().nnnz(); n++) {
    c.coo().row_indices(n)    = 0;
    c.coo().column_indices(n) = 0;
    c.coo().values(n)         = 0;
  }

  // clang-format off
  // ELL Part
  c.ell().column_indices(0,0) = 0; c.ell().values(0,0) = (value_type)1.11; 
  c.ell().column_indices(0,1) = 3; c.ell().values(0,1) = (value_type)2.22; 
  c.ell().column_indices(0,2) = 7; c.ell().values(0,2) = (value_type)3.33; 

  c.ell().column_indices(1,0) = 1; c.ell().values(1,0) = (value_type)5.55; 
  c.ell().column_indices(1,1) = 4; c.ell().values(1,1) = (value_type)6.66; 
  c.ell().column_indices(1,2) = 7; c.ell().values(1,2) = (value_type)7.77; 

  c.ell().column_indices(2,0) = 2; c.ell().values(2,0) = (value_type)9.99; 
  c.ell().column_indices(2,1) = 5; c.ell().values(2,1) = (value_type)10.10; 
  
  c.ell().column_indices(3,0) = 0; c.ell().values(3,0) = (value_type)11.11; 
  c.ell().column_indices(3,1) = 3; c.ell().values(3,1) = (value_type)12.12;
  c.ell().column_indices(3,2) = 6; c.ell().values(3,2) = (value_type)13.13; 
  
  c.ell().column_indices(4,0) = 1; c.ell().values(4,0) = (value_type)14.14; 
  c.ell().column_indices(4,1) = 4; c.ell().values(4,1) = (value_type)15.15; 
  c.ell().column_indices(4,2) = 7; c.ell().values(4,2) = (value_type)16.16;
  
  c.ell().column_indices(5,0) = 2; c.ell().values(5,0) = (value_type)17.17; 
  c.ell().column_indices(5,1) = 5; c.ell().values(5,1) = (value_type)18.18; 
  c.ell().column_indices(5,2) = 8; c.ell().values(5,2) = (value_type)19.19;
  
  c.ell().column_indices(6,0) = 3; c.ell().values(6,0) = (value_type)20.20;
  c.ell().column_indices(6,1) = 6; c.ell().values(6,1) = (value_type)21.21; 
  c.ell().column_indices(6,2) = 9; c.ell().values(6,2) = (value_type)22.22; 
  
  c.ell().column_indices(7,0) = 0; c.ell().values(7,0) = (value_type)23.23; 
  c.ell().column_indices(7,1) = 1; c.ell().values(7,1) = (value_type)24.24;
  c.ell().column_indices(7,2) = 4; c.ell().values(7,2) = (value_type)25.25; 
  
  c.ell().column_indices(8,0) = 0; c.ell().values(8,0) = (value_type)27.27; 
  c.ell().column_indices(8,1) = 5; c.ell().values(8,1) = (value_type)28.28;
  c.ell().column_indices(8,2) = 8; c.ell().values(8,2) = (value_type)29.29; 
  
  c.ell().column_indices(9,0) = 1; c.ell().values(9,0) = (value_type)30.30; 
  c.ell().column_indices(9,1) = 6; c.ell().values(9,1) = (value_type)31.31; 
  c.ell().column_indices(9,2) = 9; c.ell().values(9,2) = (value_type)32.32;

  // COO Part
  c.coo().row_indices(0) = 0; 
  c.coo().column_indices(0) = 8; 
  c.coo().values(0) = (value_type)4.44;

  c.coo().row_indices(1) = 1; 
  c.coo().column_indices(1) = 9; 
  c.coo().values(1) = (value_type)8.88;

  c.coo().row_indices(2) = 7; 
  c.coo().column_indices(2) = 7; 
  c.coo().values(2) = (value_type)26.26;
  // clang-format on
}

/**
 * @brief Builds a sample HybMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A HybMatrix type
 * @param A The HybMatrix we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  CHECK_HYB_SIZES(c, SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HYB_ELL_NNZ,
                  SMALL_HYB_COO_NNZ, SMALL_HYB_ENTRIES_PER_ROW,
                  SMALL_MATRIX_ALIGNMENT);
  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  using size_type  = typename Container::size_type;
  using value_type = typename Container::value_type;

  for (size_type i = 0; i < c.ell().values().nrows(); i++) {
    for (size_type j = 0; j < c.ell().values().ncols(); j++) {
      c.ell().column_indices(i, j) = c.ell().invalid_index();
      c.ell().values(i, j)         = value_type(0);
    }
  }

  for (size_type n = 0; c.coo().nnnz(); n++) {
    c.coo().row_indices(n)    = 0;
    c.coo().column_indices(n) = 0;
    c.coo().values(n)         = 0;
  }

  // clang-format off
  // ELL Part
  c.ell().column_indices(0,0) = 0; c.ell().values(0,0) = (value_type)1.11; 
  c.ell().column_indices(0,1) = 3; c.ell().values(0,1) = (value_type)2.22; 
  c.ell().column_indices(0,2) = 7; c.ell().values(0,2) = (value_type)3.33; 

  c.ell().column_indices(1,0) = 1; c.ell().values(1,0) = (value_type)5.55; 
  c.ell().column_indices(1,1) = 4; c.ell().values(1,1) = (value_type)6.66; 
  c.ell().column_indices(1,2) = 7; c.ell().values(1,2) = (value_type)7.77; 

  c.ell().column_indices(2,0) = 2; c.ell().values(2,0) = (value_type)9.99; 
  c.ell().column_indices(2,1) = 5; c.ell().values(2,1) = (value_type)10.10; 
  
  c.ell().column_indices(3,0) = 0; c.ell().values(3,0) = (value_type)11.11; 
  c.ell().column_indices(3,1) = 3; c.ell().values(3,1) = (value_type)12.12;
  c.ell().column_indices(3,2) = 6; c.ell().values(3,2) = (value_type)13.13; 
  
  c.ell().column_indices(4,0) = 1; c.ell().values(4,0) = (value_type)-14.14; 
  c.ell().column_indices(4,1) = 4; c.ell().values(4,1) = (value_type)-15.15; 
  c.ell().column_indices(4,2) = 7; c.ell().values(4,2) = (value_type)16.16;
  
  c.ell().column_indices(5,0) = 2; c.ell().values(5,0) = (value_type)17.17; 
  c.ell().column_indices(5,1) = 5; c.ell().values(5,1) = (value_type)18.18; 
  c.ell().column_indices(5,2) = 8; c.ell().values(5,2) = (value_type)19.19;
  
  c.ell().column_indices(6,0) = 3; c.ell().values(6,0) = (value_type)20.20;
  c.ell().column_indices(6,1) = 6; c.ell().values(6,1) = (value_type)21.21; 
  c.ell().column_indices(6,2) = 9; c.ell().values(6,2) = (value_type)22.22; 
  
  c.ell().column_indices(7,0) = 0; c.ell().values(7,0) = (value_type)23.23; 
  c.ell().column_indices(7,1) = 1; c.ell().values(7,1) = (value_type)24.24;
  c.ell().column_indices(7,2) = 4; c.ell().values(7,2) = (value_type)-25.25; 
  
  c.ell().column_indices(8,0) = 0; c.ell().values(8,0) = (value_type)27.27; 
  c.ell().column_indices(8,1) = 5; c.ell().values(8,1) = (value_type)28.28;
  c.ell().column_indices(8,2) = 8; c.ell().values(8,2) = (value_type)29.29; 
  
  c.ell().column_indices(9,0) = 1; c.ell().values(9,0) = (value_type)30.30; 
  c.ell().column_indices(9,1) = 6; c.ell().values(9,1) = (value_type)31.31; 
  c.ell().column_indices(9,2) = 9; c.ell().values(9,2) = (value_type)32.32;

  // COO Part
  c.coo().row_indices(0) = 0; 
  c.coo().column_indices(0) = 8; 
  c.coo().values(0) = (value_type)-4.44;

  c.coo().row_indices(1) = 1; 
  c.coo().column_indices(1) = 9; 
  c.coo().values(1) = (value_type)-8.88;

  c.coo().row_indices(2) = 7; 
  c.coo().column_indices(2) = 7; 
  c.coo().values(2) = (value_type)26.26;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HYB_ELL_NNZ,
           SMALL_HYB_COO_NNZ, SMALL_HYB_ENTRIES_PER_ROW,
           SMALL_MATRIX_ALIGNMENT);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container1> &&
        Morpheus::is_hyb_matrix_format_container_v<Container2>>* = nullptr) {
  bool same_nrows      = c1.nrows() == c2.nrows() ? true : false;
  bool same_ncols      = c1.ncols() == c2.ncols() ? true : false;
  bool same_nnnz       = c1.nnnz() == c2.nnnz() ? true : false;
  bool same_nalignment = c1.alignment() == c2.alignment() ? true : false;
  bool same_ell        = is_same_size(c1.ell(), c2.ell());
  bool same_coo        = is_same_size(c1.coo(), c2.coo());

  return same_nrows && same_ncols && same_nnnz && same_nalignment && same_ell &&
         same_coo;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  bool ell_is_empty = false, coo_is_empty = false;

  if (is_empty_container(c.ell())) {
    ell_is_empty = true;
  }

  if (is_empty_container(c.coo())) {
    coo_is_empty = true;
  }

  return ell_is_empty && coo_is_empty;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::HybMatrix<ValueType, IndexType, ArrayLayout, Space> create_container(
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::HybMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Container1> &&
        Morpheus::is_hyb_matrix_format_container_v<Container2>>* = nullptr) {
  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  bool same_ell_data = false, same_coo_data = false;
  if (have_same_data(c1_h.ell(), c2_h.ell())) {
    same_ell_data = true;
  }

  if (have_same_data(c1_h.coo(), c2_h.coo())) {
    same_coo_data = true;
  }

  return same_ell_data && same_coo_data;
}

}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_HYBMATRIX_HPP