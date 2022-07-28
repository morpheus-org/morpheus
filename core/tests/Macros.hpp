/**
 * Macros.hpp
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

#ifndef TEST_CORE_MACROS_HPP
#define TEST_CORE_MACROS_HPP

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
#define VALIDATE_COO_CONTAINER(A, Aref, nnnz, type)           \
  {                                                           \
    for (type n = 0; n < nnnz; n++) {                         \
      EXPECT_EQ(A.row_indices(n), Aref.row_indices(n));       \
      EXPECT_EQ(A.column_indices(n), Aref.column_indices(n)); \
      EXPECT_EQ(A.values(n), Aref.values(n));                 \
    }                                                         \
  }

/**
 * @brief Builds a sample CooMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A CooMatrix type
 * @param A The CooMatrix we will be initializing.
 */
template <typename Matrix>
void build_coomatrix(Matrix& A) {
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_COO_SIZES(A, 3, 3, 4);

  // clang-format off
  A.row_indices(0) = 0; A.column_indices(0) = 0; A.values(0) = 1.11;
  A.row_indices(1) = 0; A.column_indices(1) = 2; A.values(1) = 2.22;
  A.row_indices(2) = 1; A.column_indices(2) = 2; A.values(2) = 3.33;
  A.row_indices(3) = 2; A.column_indices(3) = 1; A.values(3) = 4.44;
  // clang-format on
}

/**
 * @brief Checks the sizes of a CsrMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_CSR_SIZES(A, num_rows, num_cols, num_nnz) \
  {                                                     \
    EXPECT_EQ(A.nrows(), num_rows);                     \
    EXPECT_EQ(A.ncols(), num_cols);                     \
    EXPECT_EQ(A.nnnz(), num_nnz);                       \
    EXPECT_EQ(A.row_offsets().size(), num_rows + 1);    \
    EXPECT_EQ(A.column_indices().size(), num_nnz);      \
    EXPECT_EQ(A.values().size(), num_nnz);              \
  }

/**
 * @brief Checks the sizes of an empty CsrMatrix container
 *
 */
#define CHECK_CSR_EMPTY(A)                   \
  {                                          \
    EXPECT_EQ(A.nrows(), 0);                 \
    EXPECT_EQ(A.ncols(), 0);                 \
    EXPECT_EQ(A.nnnz(), 0);                  \
    EXPECT_EQ(A.row_offsets().size(), 0);    \
    EXPECT_EQ(A.column_indices().size(), 0); \
    EXPECT_EQ(A.values().size(), 0);         \
  }

/**
 * @brief Checks the sizes of two CsrMatrix containers if they match
 *
 */
#define CHECK_CSR_CONTAINERS(A, B)                                   \
  {                                                                  \
    EXPECT_EQ(A.nrows(), B.nrows());                                 \
    EXPECT_EQ(A.ncols(), B.ncols());                                 \
    EXPECT_EQ(A.nnnz(), B.nnnz());                                   \
    EXPECT_EQ(A.row_offsets().size(), B.row_offsets().size());       \
    EXPECT_EQ(A.column_indices().size(), B.column_indices().size()); \
    EXPECT_EQ(A.values().size(), B.values().size());                 \
  }

/**
 * @brief Checks if the data arrays of two CsrMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_CSR_CONTAINER(A, Aref, nrows, nnnz, type)    \
  {                                                           \
    for (type n = 0; n < nrows + 1; n++) {                    \
      EXPECT_EQ(A.row_offsets(n), Aref.row_offsets(n));       \
    }                                                         \
    for (type n = 0; n < nnnz; n++) {                         \
      EXPECT_EQ(A.column_indices(n), Aref.column_indices(n)); \
      EXPECT_EQ(A.values(n), Aref.values(n));                 \
    }                                                         \
  }

/**
 * @brief Builds a sample CooMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A CooMatrix type
 * @param A The CooMatrix we will be initializing.
 */
template <typename Matrix>
void build_csrmatrix(Matrix& A) {
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_CSR_SIZES(A, 3, 3, 4);

  // clang-format off
  A.row_offsets(0) = 0; 
  A.row_offsets(1) = 2; 
  A.row_offsets(2) = 3; 
  A.row_offsets(3) = 4; 

  A.column_indices(0) = 0; A.values(0) = 1.11;
  A.column_indices(1) = 2; A.values(1) = 2.22;
  A.column_indices(2) = 2; A.values(2) = 3.33;
  A.column_indices(3) = 1; A.values(3) = 4.44;
  // clang-format on
}

#endif  // TEST_CORE_MACROS_HPP