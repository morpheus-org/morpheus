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
 * @brief Checks the sizes of a DynamicMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_DYNAMIC_SIZES(A, num_rows, num_cols, num_nnz, active_index) \
  {                                                                      \
    EXPECT_EQ(A.nrows(), num_rows);                                      \
    EXPECT_EQ(A.ncols(), num_cols);                                      \
    EXPECT_EQ(A.nnnz(), num_nnz);                                        \
    EXPECT_EQ(A.formats().index(), active_index);                    \
  }

/**
 * @brief Checks the sizes of an empty DynamicMatrix container
 *
 */
#define CHECK_DYNAMIC_EMPTY(A)           \
  {                                      \
    EXPECT_EQ(A.nrows(), 0);             \
    EXPECT_EQ(A.ncols(), 0);             \
    EXPECT_EQ(A.nnnz(), 0);              \
    EXPECT_EQ(A.formats().index(), 0);   \
  }

/**
 * @brief Checks the sizes of two DynamicMatrix containers if they match
 *
 */
#define CHECK_DYNAMIC_CONTAINERS(A, B)                      \
  {                                                         \
    EXPECT_EQ(A.nrows(), B.nrows());                        \
    EXPECT_EQ(A.ncols(), B.ncols());                        \
    EXPECT_EQ(A.nnnz(), B.nnnz());                          \
    EXPECT_EQ(A.formats().index(), B.formats().index());    \
  }

#endif  // TEST_CORE_UTILS_MACROS_DYNAMICMATRIX_HPP