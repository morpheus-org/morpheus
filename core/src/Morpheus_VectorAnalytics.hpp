/**
 * Morpheus_VectorAnalytics.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_VECTORANALYTICS_HPP
#define MORPHEUS_VECTORANALYTICS_HPP

#include <Morpheus_FormatTraits.hpp>

#include <impl/Morpheus_VectorAnalytics_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup analytics Analytics
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Finds the maximum element in the vector.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector The type of the input vector
 * @param vec The input vector
 * @param size The size of the input vector
 * @return Vector::value_type The maximum value in the vector
 */
template <typename ExecSpace, typename Vector>
typename Vector::value_type max(const Vector& vec,
                                typename Vector::size_type size) {
  static_assert(Morpheus::is_vector_container_v<Vector>,
                "The type Vector must be a valid Vector container.");
  return Impl::max<ExecSpace>(vec, size);
}

/**
 * @brief Finds the minimum element in the vector.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector The type of the input vector
 * @param vec The input vector
 * @param size The size of the input vector
 * @return Vector::value_type The minimum value in the vector
 */
template <typename ExecSpace, typename Vector>
typename Vector::value_type min(const Vector& vec,
                                typename Vector::size_type size) {
  static_assert(Morpheus::is_vector_container_v<Vector>,
                "The type Vector must be a valid Vector container.");
  return Impl::min<ExecSpace>(vec, size);
}

/**
 * @brief Finds the standard deviation of the elements in the vector around a
 * mean value.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam Vector The type of the input vector
 * @param vec The input vector
 * @param size The size of the input vector
 * @param mean The mean value
 * @return Vector::value_type The minimum value in the vector
 */
template <typename ExecSpace, typename Vector>
typename Vector::value_type std(const Vector& vec,
                                typename Vector::size_type size,
                                typename Vector::value_type mean) {
  static_assert(Morpheus::is_vector_container_v<Vector>,
                "The type Vector must be a valid Vector container.");
  return Impl::std<ExecSpace>(vec, size, mean);
}

/*! \}  // end of analytics group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_VECTORANALYTICS_HPP