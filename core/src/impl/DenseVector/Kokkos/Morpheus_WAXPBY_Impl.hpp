/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP

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
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2, Vector3>>* =
        nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using size_type       = typename Vector1::size_type;
  using range_policy    = Kokkos::RangePolicy<size_type, execution_space>;

  assert(x.size() >= n);
  assert(y.size() >= n);
  assert(w.size() >= n);

  const typename Vector1::value_array_type x_view = x.const_view();
  const typename Vector2::value_array_type y_view = y.const_view();
  typename Vector3::value_array_type w_view       = w.view();
  range_policy policy(0, n);

  if (alpha == 1.0) {
    Kokkos::parallel_for(
        "waxpby_alpha", policy, KOKKOS_LAMBDA(const size_type& i) {
          w_view[i] = x_view[i] + beta * y_view[i];
        });
  } else if (beta == 1.0) {
    Kokkos::parallel_for(
        "waxpby_beta", policy, KOKKOS_LAMBDA(const size_type& i) {
          w_view[i] = alpha * x_view[i] + y_view[i];
        });
  } else {
    Kokkos::parallel_for(
        "waxpby", policy, KOKKOS_LAMBDA(const size_type& i) {
          w_view[i] = alpha * x_view[i] + beta * y_view[i];
        });
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP