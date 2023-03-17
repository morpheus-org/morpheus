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

#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_VectorAnalytics.hpp>

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
 * @return double The average number of non-zeros
 */
template <typename Matrix>
double average_nnnz(const Matrix& A) {
  return A.nnnz() / (double)A.nrows();
}

/**
 * @brief Measures the density of the matrix.
 *
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return double The density of the matrix
 */
template <typename Matrix>
double density(const Matrix& A) {
  return A.nnnz() / (double)(A.nrows() * A.ncols());
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

/**
 * @brief Finds the maximum number of non-zeros in a row of the Matrix.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The maximum number of non-zeros in a row
 */
template <typename ExecSpace, typename Matrix>
typename Matrix::size_type max_nnnz(const Matrix& A) {
  static_assert(Morpheus::is_matrix_container_v<Matrix>,
                "The type Matrix must be a valid matrix container.");
  using value_type   = typename Matrix::index_type;
  using index_type   = typename Matrix::size_type;
  using array_layout = typename Matrix::array_layout;
  using backend      = typename Matrix::backend;
  using Vector =
      Morpheus::DenseVector<value_type, index_type, array_layout, backend>;

  Vector nnnz_per_row(A.nrows(), 0);
  Impl::count_nnz_per_row<ExecSpace>(A, nnnz_per_row, false);

  return Morpheus::max<ExecSpace>(nnnz_per_row, nnnz_per_row.size());
}

/**
 * @brief Finds the minimum number of non-zeros in a row of the Matrix.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The minimum number of non-zeros in a row
 */
template <typename ExecSpace, typename Matrix>
typename Matrix::size_type min_nnnz(const Matrix& A) {
  static_assert(Morpheus::is_matrix_container_v<Matrix>,
                "The type Matrix must be a valid matrix container.");
  using value_type   = typename Matrix::index_type;
  using index_type   = typename Matrix::size_type;
  using array_layout = typename Matrix::array_layout;
  using backend      = typename Matrix::backend;
  using Vector =
      Morpheus::DenseVector<value_type, index_type, array_layout, backend>;

  Vector nnnz_per_row(A.nrows(), 0);
  Impl::count_nnz_per_row<ExecSpace>(A, nnnz_per_row, false);

  return Morpheus::min<ExecSpace>(nnnz_per_row, nnnz_per_row.size());
}

/**
 * @brief Finds the standard deviation around a mean of non-zeros in the Matrix.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return double The standard deviation around a mean of non-zeros in the
 * matrix
 */
template <typename ExecSpace, typename Matrix>
typename Matrix::size_type std_nnnz(const Matrix& A) {
  static_assert(Morpheus::is_matrix_container_v<Matrix>,
                "The type Matrix must be a valid matrix container.");
  using value_type   = typename Matrix::index_type;
  using index_type   = typename Matrix::size_type;
  using array_layout = typename Matrix::array_layout;
  using backend      = typename Matrix::backend;
  using Vector =
      Morpheus::DenseVector<value_type, index_type, array_layout, backend>;

  Vector nnnz_per_row(A.nrows(), 0);
  Impl::count_nnz_per_row<ExecSpace>(A, nnnz_per_row, false);

  return Morpheus::std<ExecSpace>(nnnz_per_row, nnnz_per_row.size(),
                                  A.nnnz() / A.nrows());
}

/**
 * @brief Counts the number of the diagonals of the matrix with at least one
 * non-zero.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @param nnz_per_diagonal A vector containing the number of non-zeros per
 * diagonal
 * @param init Whether to initialize nnz_per_row to zero.
 */
template <typename ExecSpace, typename Matrix, typename Vector>
void count_nnz_per_diagonal(const Matrix& A, Vector& nnz_per_diagonal,
                            const bool init = true) {
  Impl::count_nnz_per_diagonal<ExecSpace>(A, nnz_per_diagonal, init);
}

/**
 * @brief Counts the number of diagonals with at least one non-zero entry.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The number of diagonals
 */
template <typename ExecSpace, typename Matrix>
typename Matrix::size_type count_diagonals(const Matrix& A) {
  static_assert(Morpheus::is_matrix_container_v<Matrix>,
                "The type Matrix must be a valid matrix container.");
  using value_type   = typename Matrix::index_type;
  using index_type   = typename Matrix::size_type;
  using array_layout = typename Matrix::array_layout;
  using backend      = typename Matrix::backend;
  using Vector =
      Morpheus::DenseVector<value_type, index_type, array_layout, backend>;

  Vector nnnz_per_diag(A.nrows() + A.ncols() - 1, 0);
  Impl::count_nnz_per_diagonal<ExecSpace>(A, nnnz_per_diag, false);

  return Morpheus::count_nnz<ExecSpace>(nnnz_per_diag);
}

/**
 * @brief Counts the number of true diagonals.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the input matrix
 * @param A The input matrix
 * @return Matrix::size_type The number of true diagonals
 */
template <typename ExecSpace, typename Matrix>
typename Matrix::size_type count_true_diagonals(
    const Matrix& A, typename Matrix::index_type threshold) {
  static_assert(Morpheus::is_matrix_container_v<Matrix>,
                "The type Matrix must be a valid matrix container.");
  using value_type   = typename Matrix::index_type;
  using index_type   = typename Matrix::size_type;
  using array_layout = typename Matrix::array_layout;
  using backend      = typename Matrix::backend;
  using Vector =
      Morpheus::DenseVector<value_type, index_type, array_layout, backend>;

  Vector nnnz_per_diag(A.nrows() + A.ncols() - 1, 0);
  Impl::count_nnz_per_diagonal<ExecSpace>(A, nnnz_per_diag, false);

  return Morpheus::count_nnz<ExecSpace>(nnnz_per_diag, threshold);
}

/*! \}  // end of analytics group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXANALYTICS_HPP
