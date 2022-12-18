/**
 * Morpheus_Reduction.hpp
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

#ifndef MORPHEUS_REDUCTION_HPP
#define MORPHEUS_REDUCTION_HPP

#include <impl/Morpheus_Reduction_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup dense_vector_algorithms DenseVector Algorithms
 * \ingroup algorithms
 * \{
 *
 */

/**
 * @brief Performs a sum reduction on the contents of a vector.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector Type of input vector
 * @param in The input vector
 * @param size The number of vector elements to run the operation for
 * @return Vector::value_type Scalar value of the result
 */
template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(const Vector& in,
                                   typename Vector::size_type size) {
  static_assert(is_dense_vector_format_container_v<Vector>,
                "in must be a DenseVector container");
  return Impl::reduce<ExecSpace>(in, size);
}
/*! \}  // end of dense_vector_algorithms group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_REDUCTION_HPP