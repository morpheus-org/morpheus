/**
 * Morpheus_MatrixAnalytics.hpp
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

#ifndef MORPHEUS_MATRIXANALYTICS_HPP
#define MORPHEUS_MATRIXANALYTICS_HPP

#include <impl/Morpheus_MatrixAnalytics_Impl.hpp>
#include <impl/Dynamic/Morpheus_MatrixAnalytics_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup analytics Analytics
 * \brief Analytics based on the input container.
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Counts the number of rows in a matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The number of rows
 */
template <typename Matrix>
typename Matrix::size_type number_of_rows(const Matrix& A) {
  return A.nrows();
}

/**
 * @brief Counts the number of columns in a matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The number of columns
 */
template <typename Matrix>
typename Matrix::size_type number_of_columns(const Matrix& A) {
  return A.ncols();
}

/**
 * @brief Counts the number of non-zeros in a matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The number of non-zeros
 */
template <typename Matrix>
typename Matrix::size_type number_of_nnz(const Matrix& A) {
  return A.nnnz();
}

/**
 * @brief Counts the average number of non-zeros in a matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The average number of non-zeros
 */
template <typename Matrix>
typename Matrix::size_type average_nnnz(const Matrix& A) {
  return A.nnnz() / A.nrows();
}

/**
 * @brief Measures the density of the matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The density of the matrix
 */
template <typename Matrix>
typename Matrix::size_type density(const Matrix& A) {
  return A.nnnz() / (A.nrows() * A.ncols());
}

/**
 * @brief Counts the number of non-zeros per row in the matrix.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @tparam Vector The type of the output vector
 * @param A The known matrix
 * @param nnz_per_row A vector containing the number of non-zeros per row
 * @param init Whether to initialize nnz_per_row to zero.
 */
template <typename ExecSpace, typename Matrix, typename Vector>
void count_nnz_per_row(const Matrix& A, Vector& nnz_per_row,
                       const bool init = true) {
  Impl::count_nnz_per_row<ExecSpace>(A, nnz_per_row, init);
}

/*! \}  // end of analytics group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXANALYTICS_HPP
