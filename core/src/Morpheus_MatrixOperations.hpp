/**
 * Morpheus_MatrixOperations.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_MATRIXOPERATIONS_HPP
#define MORPHEUS_MATRIXOPERATIONS_HPP

#include <impl/Morpheus_MatrixOperations_Impl.hpp>
#include <impl/Dynamic/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {

/**
 * Updates the main diagonal of the matrix with contents of the diagonal vector.
 *
 * \tparam ExecSpace
 * \tparam Matrix
 * \tparam Vector
 *
 * \param A The matrix
 * \param diagonal The matrix diagonal represented as a vector
 *
 * \note The sparsity pattern of the matrix remains unchanged i.e it only
 * updates the non-zero elements on the main diagonal.
 *
 */
template <typename ExecSpace, typename Matrix, typename Vector>
void update_diagonal(Matrix& A, const Vector& diagonal) {
  Impl::update_diagonal<ExecSpace>(A, diagonal);
}

/**
 * Gets the main diagonal of the matrix and places it in a vector.
 *
 * \tparam ExecSpace
 * \tparam Matrix
 * \tparam Vector
 *
 * \param A The matrix
 * \param diagonal The main matrix diagonal represented as a vector
 *
 */
template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(const Matrix& A, Vector& diagonal) {
  Impl::get_diagonal<ExecSpace>(A, diagonal);
}

/**
 * Set a single entry into a matrix.
 *
 * \tparam ExecSpace
 * \tparam Matrix
 * \tparam IndexType
 * \tparam ValueType
 *
 * \param A The matrix
 * \param row The row location of the entry
 * \param col The column location of the entry
 * \param value The value to insert
 */
template <typename ExecSpace, typename Matrix, typename IndexType,
          typename ValueType>
void set_value(Matrix& A, IndexType row, IndexType col, ValueType value) {
  Impl::set_value(A, row, col, value);
}

/**
 * Inserts or adds a block of values into a matrix.
 *
 * \tparam ExecSpace
 * \tparam Matrix
 * \tparam IndexVector
 * \tparam ValueVector
 *
 * \param A The matrix
 * \param m The number of rows
 * \param idxm Row global indices
 * \param n The number of columns
 * \param idxn Column global indices
 * \param values A logically two-dimensional array of values
 */
template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
void set_values(Matrix& A, typename IndexVector::value_type m,
                const IndexVector idxm, typename IndexVector::value_type n,
                const IndexVector idxn, ValueVector values) {
  Impl::set_values<ExecSpace>(A, m, idxm, n, idxn, values);
}

/**
 * Computes the transpose of the given matrix.
 *
 * \tparam ExecSpace
 * \tparam Matrix
 * \tparam TransposeMatrix
 *
 * \param A The matrix
 * \param B The transposed matrix
 */
template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(const Matrix& A, TransposeMatrix& At) {
  Impl::transpose<ExecSpace>(A, At);
}

}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXOPERATIONS_HPP
