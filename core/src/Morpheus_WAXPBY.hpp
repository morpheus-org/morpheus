/**
 * Morpheus_WAXPBY.hpp
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

#ifndef MORPHEUS_WAXPBY_HPP
#define MORPHEUS_WAXPBY_HPP

#include <impl/Morpheus_WAXPBY_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup dense_vector_algorithms DenseVector Algorithms
 * \ingroup algorithms
 * \{
 *
 */

/**
 * @brief Computes the update of a vector with the sum of two scaled vectors
 * where: w = alpha*x + beta*y
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector1 Type of vector x
 * @tparam Vector2 Type of vector y
 * @tparam Vector3 Type of vector w
 * @param n The number of vector elements to run the operation for
 * @param alpha Scalar applied to x
 * @param x The first input vector
 * @param beta Scalar applied to y
 * @param y The second input vector
 * @param w The output vector
 */
template <typename ExecSpace, typename Vector1, typename Vector2,
          typename Vector3>
inline void waxpby(const size_t n, const typename Vector1::value_type alpha,
                   const Vector1& x, const typename Vector2::value_type beta,
                   const Vector2& y, Vector3& w) {
  static_assert(is_dense_vector_format_container_v<Vector1>,
                "x must be a DenseVector container");
  static_assert(is_dense_vector_format_container_v<Vector2>,
                "y must be a DenseVector container");
  static_assert(is_dense_vector_format_container_v<Vector3>,
                "w must be a DenseVector container");
  Impl::waxpby<ExecSpace>(n, alpha, x, beta, y, w);
}
/*! \}  // end of dense_vector_algorithms group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_WAXPBY_HPP