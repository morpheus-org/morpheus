/**
 * Morpheus_Multiply.hpp
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

#ifndef MORPHEUS_MULTIPLY_HPP
#define MORPHEUS_MULTIPLY_HPP

#include <Morpheus_AlgorithmTags.hpp>
#include <impl/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {

template <typename ExecSpace, typename Algorithm, typename LinearOperator,
          typename MatrixOrVector1, typename MatrixOrVector2>
inline void multiply(const LinearOperator& A, const MatrixOrVector1& x,
                     MatrixOrVector2& y) {
  Impl::multiply<ExecSpace>(A, x, y, typename LinearOperator::tag{},
                            typename MatrixOrVector1::tag{},
                            typename MatrixOrVector2::tag{}, Algorithm{});
}

// Default algorithm to run with multiply is always Alg0
template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(const LinearOperator& A, const MatrixOrVector1& x,
                     MatrixOrVector2& y) {
  Impl::multiply<ExecSpace>(A, x, y, typename LinearOperator::tag{},
                            typename MatrixOrVector1::tag{},
                            typename MatrixOrVector2::tag{}, Alg0{});
}

}  // namespace Morpheus

#endif  // MORPHEUS_MULTIPLY_HPP