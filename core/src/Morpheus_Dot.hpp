/**
 * Morpheus_Dot.hpp
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

#ifndef MORPHEUS_DOT_HPP
#define MORPHEUS_DOT_HPP

#include <impl/Morpheus_Dot_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup dense_vector_algorithms DenseVector Algorithms
 * \brief Algorithms for the DenseVector container.
 * \ingroup algorithms
 * \{
 *
 */

/**
 * @brief Computes the dot product of two vectors.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector1 Type of vector x
 * @tparam Vector2 Type of vector y
 * @param n The number of vector elements to run the operation for
 * @param x The first input vector
 * @param y The second input vector
 * @return Vector2::value_type Scalar value of the result
 *
 */
template <typename ExecSpace, typename Vector1, typename Vector2>
inline typename Vector2::value_type dot(typename Vector1::size_type n,
                                        const Vector1& x, const Vector2& y) {
  static_assert(is_dense_vector_format_container_v<Vector1>,
                "x must be a DenseVector container");
  static_assert(is_dense_vector_format_container_v<Vector2>,
                "y must be a DenseVector container");
  return Impl::dot<ExecSpace>(n, x, y);
}
/*! \}  // end of dense_vector_algorithms group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_DOT_HPP