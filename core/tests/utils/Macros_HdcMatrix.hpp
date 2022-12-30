/**
 * Macros_HdcMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_HDCMATRIX_HPP
#define TEST_CORE_UTILS_MACROS_HDCMATRIX_HPP

#include <Morpheus_Core.hpp>

#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>

/**
 * @brief Checks the sizes of a HdcMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_HDC_SIZES(A, num_rows, num_cols, num_dia_entries,      \
                        num_csr_entries, num_entries_per_row, align) \
  {                                                                  \
    EXPECT_EQ(A.nrows(), num_rows);                                  \
    EXPECT_EQ(A.ncols(), num_cols);                                  \
    EXPECT_EQ(A.nnnz(), num_dia_entries + num_csr_entries);          \
    EXPECT_EQ(A.alignment(), align);                                 \
    CHECK_DIA_SIZES(A.dia(), num_rows, num_cols, num_dia_entries,    \
                    num_entries_per_row, align);                     \
    CHECK_CSR_SIZES(A.csr(), num_rows, num_cols, num_csr_entries);   \
  }

/**
 * @brief Checks the sizes of an empty HdcMatrix container
 *
 */
#define CHECK_HDC_EMPTY(A)       \
  {                              \
    EXPECT_EQ(A.nrows(), 0);     \
    EXPECT_EQ(A.ncols(), 0);     \
    EXPECT_EQ(A.nnnz(), 0);      \
    EXPECT_EQ(A.alignment(), 0); \
    CHECK_DIA_EMPTY(A.dia());    \
    CHECK_CSR_EMPTY(A.csr());    \
  }

/**
 * @brief Checks the sizes of two HdcMatrix containers if they match
 *
 */
#define CHECK_HDC_CONTAINERS(A, B)           \
  {                                          \
    EXPECT_EQ(A.nrows(), B.nrows());         \
    EXPECT_EQ(A.ncols(), B.ncols());         \
    EXPECT_EQ(A.nnnz(), B.nnnz());           \
    EXPECT_EQ(A.alignment(), B.alignment()); \
    CHECK_DIA_CONTAINERS(A.dia(), B.dia());  \
    CHECK_CSR_CONTAINERS(A.csr(), B.csr());  \
  }

/**
 * @brief Checks if the data arrays of two HdcMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_HDC_CONTAINER(A, Aref)                          \
  {                                                              \
    VALIDATE_DIA_CONTAINER(A.dia(), Aref.dia());                 \
    VALIDATE_CSR_CONTAINER(A.csr(), Aref.csr(), A.csr().nrows(), \
                           A.csr().nnnz());                      \
  }

namespace Morpheus {
namespace Test {
template <typename Container>
void reset_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;

  // clang-format off
  // DIA Part
  c.dia().diagonal_offsets(0) = -3; 
  c.dia().diagonal_offsets(1) = 0; 
  c.dia().diagonal_offsets(2) = 3; 

  c.dia().values(0, 0) = (value_type)0;     c.dia().values(0, 1) = (value_type)1.11;  c.dia().values(0, 2) = (value_type)2.22;
  c.dia().values(1, 0) = (value_type)0;     c.dia().values(1, 1) = (value_type)5.55;  c.dia().values(1, 2) = (value_type)6.66;
  c.dia().values(2, 0) = (value_type)0;     c.dia().values(2, 1) = (value_type)9.99;  c.dia().values(2, 2) = (value_type)10.10;
  c.dia().values(3, 0) = (value_type)11.11; c.dia().values(3, 1) = (value_type)12.12; c.dia().values(3, 2) = (value_type)13.13;
  c.dia().values(4, 0) = (value_type)14.14; c.dia().values(4, 1) = (value_type)15.15; c.dia().values(4, 2) = (value_type)16.16;
  c.dia().values(5, 0) = (value_type)17.17; c.dia().values(5, 1) = (value_type)18.18; c.dia().values(5, 2) = (value_type)19.19;
  c.dia().values(6, 0) = (value_type)20.20; c.dia().values(6, 1) = (value_type)21.21; c.dia().values(6, 2) = (value_type)22.22;
  c.dia().values(7, 0) = (value_type)25.25; c.dia().values(7, 1) = (value_type)26.26; c.dia().values(7, 2) = (value_type)0;
  c.dia().values(8, 0) = (value_type)28.28; c.dia().values(8, 1) = (value_type)29.29; c.dia().values(8, 2) = (value_type)0;
  c.dia().values(9, 0) = (value_type)31.31; c.dia().values(9, 1) = (value_type)32.32; c.dia().values(9, 2) = (value_type)0;

  // CSR Part
  c.csr().row_offsets(0)  = 0; 
  c.csr().row_offsets(1)  = 2; 
  c.csr().row_offsets(2)  = 4; 
  c.csr().row_offsets(3)  = 4;
  c.csr().row_offsets(4)  = 4; 
  c.csr().row_offsets(5)  = 4; 
  c.csr().row_offsets(6)  = 4; 
  c.csr().row_offsets(7)  = 4;
  c.csr().row_offsets(8)  = 6; 
  c.csr().row_offsets(9)  = 7; 
  c.csr().row_offsets(10) = 8; 

  c.csr().column_indices(0)  = 7; c.csr().values(0)  = (value_type)3.33;
  c.csr().column_indices(1)  = 8; c.csr().values(1)  = (value_type)4.44;

  c.csr().column_indices(2)  = 7; c.csr().values(2)  = (value_type)7.77;
  c.csr().column_indices(3)  = 9; c.csr().values(3)  = (value_type)8.88;

  c.csr().column_indices(4) = 0; c.csr().values(4) = (value_type)23.23;
  c.csr().column_indices(5) = 1; c.csr().values(5) = (value_type)24.24;

  c.csr().column_indices(6) = 0; c.csr().values(6) = (value_type)27.27;

  c.csr().column_indices(7) = 1; c.csr().values(7) = (value_type)30.30;
  // clang-format on
}

/**
 * @brief Builds a sample HdcMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A HdcMatrix type
 * @param A The HdcMatrix we will be initializing.
 */
template <typename Container>
void build_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  CHECK_HDC_SIZES(c, SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HDC_DIA_NNZ,
                  SMALL_HDC_CSR_NNZ, SMALL_HDC_DIA_NDIAG,
                  SMALL_MATRIX_ALIGNMENT);
  reset_small_container(c);
}

template <typename Container>
void update_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  using value_type = typename Container::value_type;

  // clang-format off
  // DIA Part
  c.dia().diagonal_offsets(0) = -3; 
  c.dia().diagonal_offsets(1) = 0; 
  c.dia().diagonal_offsets(2) = 3; 

  c.dia().values(0, 0) = (value_type)0;      c.dia().values(0, 1) = (value_type)1.11;   c.dia().values(0, 2) = (value_type)2.22;
  c.dia().values(1, 0) = (value_type)0;      c.dia().values(1, 1) = (value_type)5.55;   c.dia().values(1, 2) = (value_type)6.66;
  c.dia().values(2, 0) = (value_type)0;      c.dia().values(2, 1) = (value_type)9.99;   c.dia().values(2, 2) = (value_type)10.10;
  c.dia().values(3, 0) = (value_type)11.11;  c.dia().values(3, 1) = (value_type)12.12;  c.dia().values(3, 2) = (value_type)13.13;
  c.dia().values(4, 0) = (value_type)-14.14; c.dia().values(4, 1) = (value_type)-15.15; c.dia().values(4, 2) = (value_type)16.16;
  c.dia().values(5, 0) = (value_type)17.17;  c.dia().values(5, 1) = (value_type)18.18;  c.dia().values(5, 2) = (value_type)19.19;
  c.dia().values(6, 0) = (value_type)20.20;  c.dia().values(6, 1) = (value_type)21.21;  c.dia().values(6, 2) = (value_type)22.22;
  c.dia().values(7, 0) = (value_type)-25.25; c.dia().values(7, 1) = (value_type)26.26;  c.dia().values(7, 2) = (value_type)0;
  c.dia().values(8, 0) = (value_type)28.28;  c.dia().values(8, 1) = (value_type)29.29;  c.dia().values(8, 2) = (value_type)0;
  c.dia().values(9, 0) = (value_type)31.31;  c.dia().values(9, 1) = (value_type)32.32;  c.dia().values(9, 2) = (value_type)0;

  // CSR Part
  c.csr().row_offsets(0)  = 0; 
  c.csr().row_offsets(1)  = 2; 
  c.csr().row_offsets(2)  = 4; 
  c.csr().row_offsets(3)  = 4;
  c.csr().row_offsets(4)  = 4; 
  c.csr().row_offsets(5)  = 4; 
  c.csr().row_offsets(6)  = 4; 
  c.csr().row_offsets(7)  = 4;
  c.csr().row_offsets(8)  = 6; 
  c.csr().row_offsets(9)  = 7; 
  c.csr().row_offsets(10) = 8; 

  c.csr().column_indices(0)  = 7; c.csr().values(0)  = (value_type)3.33;
  c.csr().column_indices(1)  = 8; c.csr().values(1)  = (value_type)-4.44;

  c.csr().column_indices(2)  = 7; c.csr().values(2)  = (value_type)7.77;
  c.csr().column_indices(3)  = 9; c.csr().values(3)  = (value_type)-8.88;

  c.csr().column_indices(4) = 0; c.csr().values(4) = (value_type)23.23;
  c.csr().column_indices(5) = 1; c.csr().values(5) = (value_type)24.24;

  c.csr().column_indices(6) = 0; c.csr().values(6) = (value_type)27.27;

  c.csr().column_indices(7) = 1; c.csr().values(7) = (value_type)30.30;
  // clang-format on
}

template <class Container>
void setup_small_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  c.resize(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HDC_DIA_NNZ,
           SMALL_HDC_CSR_NNZ, SMALL_HDC_DIA_NDIAG, SMALL_MATRIX_ALIGNMENT);
  build_small_container(c);
}

template <class Container1, class Container2>
bool is_same_size(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container1> &&
        Morpheus::is_hdc_matrix_format_container_v<Container2>>* = nullptr) {
  bool same_nrows      = c1.nrows() == c2.nrows() ? true : false;
  bool same_ncols      = c1.ncols() == c2.ncols() ? true : false;
  bool same_nnnz       = c1.nnnz() == c2.nnnz() ? true : false;
  bool same_nalignment = c1.alignment() == c2.alignment() ? true : false;
  bool same_dia        = is_same_size(c1.dia(), c2.dia());
  bool same_csr        = is_same_size(c1.csr(), c2.csr());

  return same_nrows && same_ncols && same_nnnz && same_nalignment && same_dia &&
         same_csr;
}

template <class Container>
bool is_empty_container(
    Container& c,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  bool dia_is_empty = false, csr_is_empty = false;

  if (is_empty_container(c.dia())) {
    dia_is_empty = true;
  }

  if (is_empty_container(c.csr())) {
    csr_is_empty = true;
  }

  return dia_is_empty && csr_is_empty;
}

template <typename Container, typename ValueType, typename IndexType,
          typename ArrayLayout, typename Space>
Morpheus::HdcMatrix<ValueType, IndexType, ArrayLayout, Space> create_container(
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container>>* = nullptr) {
  return Morpheus::HdcMatrix<ValueType, IndexType, ArrayLayout, Space>();
}

template <class Container1, class Container2>
bool have_same_data(
    Container1& c1, Container2& c2,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Container1> &&
        Morpheus::is_hdc_matrix_format_container_v<Container2>>* = nullptr) {
  if (!is_same_size(c1, c2)) return false;

  typename Container1::HostMirror c1_h;
  c1_h.resize(c1);
  Morpheus::copy(c1, c1_h);

  typename Container1::HostMirror c2_h;
  c2_h.resize(c2);
  Morpheus::copy(c2, c2_h);

  bool same_dia_data = false, same_csr_data = false;
  if (have_same_data(c1_h.dia(), c2_h.dia())) {
    same_dia_data = true;
  }

  if (have_same_data(c1_h.csr(), c2_h.csr())) {
    same_csr_data = true;
  }

  return same_dia_data && same_csr_data;
}

}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_UTILS_MACROS_HDCMATRIX_HPP