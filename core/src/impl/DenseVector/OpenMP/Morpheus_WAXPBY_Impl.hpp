/**
 * Morpheus_WAXPBY_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_OPENMP_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_OPENMP_WAXPBY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <cassert>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2,
          typename Vector3>
inline void waxpby(
    const typename Vector1::size_type n,
    const typename Vector1::value_type alpha, const Vector1& x,
    const typename Vector2::value_type beta, const Vector2& y, Vector3& w,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::is_dense_vector_format_container_v<Vector3> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2, Vector3>>* =
        nullptr) {
  using size_type = typename Vector1::size_type;

  assert(x.size() >= n);
  assert(y.size() >= n);
  assert(w.size() >= n);

  if (alpha == 1.0) {
#pragma omp parallel for
    for (size_type i = 0; i < n; i++) w[i] = x[i] + beta * y[i];
  } else if (beta == 1.0) {
#pragma omp parallel for
    for (size_type i = 0; i < n; i++) w[i] = alpha * x[i] + y[i];
  } else {
#pragma omp parallel for
    for (size_type i = 0; i < n; i++) w[i] = alpha * x[i] + beta * y[i];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DENSEVECTOR_OPENMP_WAXPBY_IMPL_HPP