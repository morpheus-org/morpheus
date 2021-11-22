/**
 * Morpheus_Fwd_Multiply.hpp
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

#ifndef MORPHEUS_FWD_MULTIPLY_HPP
#define MORPHEUS_FWD_MULTIPLY_HPP

namespace Morpheus {

template <typename ExecSpace, typename Algorithm, typename Matrix,
          typename Vector1, typename Vector2>
void multiply(const Matrix& A, const Vector1& x, Vector2& y);

template <typename ExecSpace, typename Matrix, typename Vector1,
          typename Vector2>
void multiply(const Matrix& A, const Vector1& x, Vector2& y);

}  // namespace Morpheus

#endif  // MORPHEUS_FWD_MULTIPLY_HPP
