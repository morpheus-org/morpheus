/**
 * Morpheus_Fwd_MatrixOperations.hpp
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

#ifndef MORPHEUS_FWD_MATRIXOPERATIONS_HPP
#define MORPHEUS_FWD_MATRIXOPERATIONS_HPP

namespace Morpheus {

template <typename ExecSpace, typename Algorithm, typename SparseMatrix,
          typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal);

// Default algorithm to run with update_diagonal is always Alg0
template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal);

}  // namespace Morpheus

#endif  // MORPHEUS_FWD_MATRIXOPERATIONS_HPP