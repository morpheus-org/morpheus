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

// Updates the main diagonal of the matrix with contents of the diagonal vector.
// Note that it doesn't change the sparsity pattern of the matrix
// i.e it only updates the non-zero elements on the main diagonal.
template <typename ExecSpace, typename Algorithm, typename SparseMatrix,
          typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal) {
  Impl::update_diagonal<ExecSpace>(A, diagonal, typename SparseMatrix::tag{},
                                   typename Vector::tag{}, Algorithm{});
}

// Default algorithm to run with update_diagonal is always Alg0
template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal) {
  Impl::update_diagonal<ExecSpace>(A, diagonal, typename SparseMatrix::tag{},
                                   typename Vector::tag{}, Alg0{});
}

}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXOPERATIONS_HPP