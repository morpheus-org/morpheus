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

/**
 * @brief Checks the sizes of a DiaMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_DIA_SIZES(A, num_rows, num_cols, num_nnz, num_diag, alignment) \
  {                                                                          \
    EXPECT_EQ(A.nrows(), num_rows);                                          \
    EXPECT_EQ(A.ncols(), num_cols);                                          \
    EXPECT_EQ(A.nnnz(), num_nnz);                                            \
    EXPECT_EQ(A.ndiags(), num_diag);                                         \
    EXPECT_EQ(A.alignment(), alignment);                                     \
    EXPECT_EQ(A.diagonal_offsets().size(), num_diag);                        \
    EXPECT_EQ(A.values().size(), num_nnz);                                   \
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
    EXPECT_EQ(A.values().size(), 0);           \
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
    EXPECT_EQ(A.values().size(), B.values().size());                     \
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
    for (type i = 0; i < A.values.nrows(); i++) {                 \
      for (type j = 0; j < A.values.ncols(); j++) {               \
        EXPECT_EQ(A.values(i, j), Aref.values(i, j));             \
      }                                                           \
    }                                                             \
  }

/**
 * @brief Builds a sample DiaMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A DiaMatrix type
 * @param A The DiaMatrix we will be initializing.
 */
template <typename Matrix>
void build_diamatrix(Matrix& A) {
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_DIA_SIZES(A, 3, 3, 4, 4, 32);

  // clang-format off
  A.diagonal_offsets(0) = -1; 
  A.diagonal_offsets(1) = 0; 
  A.diagonal_offsets(2) = 1; 
  A.diagonal_offsets(3) = 2; 
  // values are:
  // [*    1.11 0    2.22]
  // [0    0    3.33 *]
  // [4.44 0    *    *]
  // * -> -99
  A.values(0,0) = -99; A.values(1,0) = 0; A.values(2,0) = 4.44;
  A.values(0,1) = 1.11; A.values(1,1) = 0; A.values(2,1) = 0;
  A.values(0,2) = 0; A.values(1,2) = 3.33; A.values(2,2) = -99;
  A.values(0,3) = 2.22; A.values(1,3) = -99; A.values(2,3) = -99;
  // clang-format on
}

#endif  // TEST_CORE_UTILS_MACROS_DIAMATRIX_HPP