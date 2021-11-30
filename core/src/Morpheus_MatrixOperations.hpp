/**
 * Morpheus_MatrixOperations.hpp
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

#ifndef MORPHEUS_MATRIXOPERATIONS_HPP
#define MORPHEUS_MATRIXOPERATIONS_HPP

#include <Morpheus_AlgorithmTags.hpp>
#include <impl/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {

/**
 * Updates the main diagonal of the matrix with contents of the diagonal vector.
 *
 * \tparam ExecSpace
 * \tparam SparseMatrix
 * \tparam Vector
 *
 * \param A The matrix
 * \param diagonal The diagonal matrix represented as a vector
 *
 * \note The sparsity pattern of the matrix remains unchanged i.e it only
 * updates the non-zero elements on the main diagonal.
 *
 */
template <typename ExecSpace, typename SparseMatrix, typename Vector>
void update_diagonal(SparseMatrix& A, const Vector& diagonal) {
  Impl::update_diagonal<ExecSpace>(A, diagonal, typename SparseMatrix::tag{},
                                   typename Vector::tag{});
}

/**
 * Set a single entry into a matrix.
 *
 * \tparam SparseMatrix
 * \tparam IndexType
 * \tparam ValueType
 *
 * \param A The matrix
 * \param row The row location of the entry
 * \param col The column location of the entry
 * \param value The value to insert
 */
template <typename SparseMatrix, typename IndexType, typename ValueType>
void set_value(SparseMatrix& A, IndexType row, IndexType col, ValueType value) {
  Impl::set_value(A, row, col, value);
}

/**
 * Inserts or adds a block of values into a matrix.
 *
 * \tparam ExecSpace
 * \tparam SparseMatrix
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
template <typename ExecSpace, typename SparseMatrix, typename IndexVector,
          typename ValueVector>
void set_values(SparseMatrix& A, typename IndexVector::value_type m,
                const IndexVector idxm, typename IndexVector::value_type n,
                const IndexVector idxn, ValueVector values) {
  Impl::set_values<ExecSpace>(A, m, idxm, n, idxn, values);
}

}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXOPERATIONS_HPP
