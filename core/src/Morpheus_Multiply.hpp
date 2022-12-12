/**
 * Morpheus_Multiply.hpp
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

#ifndef MORPHEUS_MULTIPLY_HPP
#define MORPHEUS_MULTIPLY_HPP

#include <impl/Morpheus_Multiply_Impl.hpp>
#include <impl/Dynamic/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
/**
 * \defgroup algorithms Algorithms
 * \par Overview
 * TODO
 *
 */

/**
 * \addtogroup sparse_algorithms Sparse Algorithms
 * \brief Algorithms for sparse and dynamic containers
 * \ingroup algorithms
 * \{
 *
 */

/**
 * @brief Computes the matrix vector product y = Ax
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the system matrix
 * @tparam Vector The type of the vectors
 * @param A The known system matrix
 * @param x The known vector
 * @param y The vector that contains the result Ax
 * @param init Whether to initialize y to zero.
 */
template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(const Matrix& A, const Vector& x, Vector& y,
                     const bool init) {
  Impl::multiply<ExecSpace>(A, x, y, init);
}

/**
 * @brief Computes the matrix vector product y = Ax
 *
 * @tparam ExecSpace ExecSpace Execution space to run the algorithm
 * @tparam Matrix The type of the system matrix
 * @tparam Vector The type of the vectors
 * @param A The known system matrix
 * @param x The known vector
 * @param y The vector that contains the result Ax
 *
 * \note This routine initializes the vector y to zero by default.
 */
template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(const Matrix& A, const Vector& x, Vector& y) {
  Impl::multiply<ExecSpace>(A, x, y, true);
}
/*! \}  // end of sparse_algorithms group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_MULTIPLY_HPP