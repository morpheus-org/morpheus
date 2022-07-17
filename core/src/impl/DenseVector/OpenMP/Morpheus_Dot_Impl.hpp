/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_OPENMP_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_OPENMP_DOT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2>
inline typename Vector1::value_type dot(
    typename Vector1::index_type n, const Vector1& x, const Vector2& y,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector1,
                               Vector2>>* = nullptr) {
  using index_type = typename Vector1::index_type;
  using value_type = typename Vector1::non_const_value_type;

  value_type result = value_type(0);

  if (y.data() == x.data()) {
#pragma omp parallel for reduction(+ : result)
    for (index_type i = 0; i < n; i++) result += x[i] * x[i];
  } else {
#pragma omp parallel for reduction(+ : result)
    for (index_type i = 0; i < n; i++) result += x[i] * y[i];
  }

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DENSEVECTOR_OPENMP_DOT_IMPL_HPP