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

#ifndef MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
inline void waxpby(const typename Vector::index_type n,
                   const typename Vector::value_type alpha, const Vector& x,
                   const typename Vector::value_type beta, const Vector& y,
                   Vector& w,
                   typename std::enable_if_t<
                       Morpheus::is_dense_vector_format_container_v<Vector> &&
                       Morpheus::is_generic_backend_v<ExecSpace> &&
                       Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using execution_space = ExecSpace;
  using IndexType       = Kokkos::IndexType<typename Vector::index_type>;
  using range_policy    = Kokkos::RangePolicy<IndexType, execution_space>;
  using value_array     = typename Vector::value_array_type;
  using index_type      = typename Vector::index_type;

  const value_array x_view = x.const_view(), y_view = y.const_view();
  value_array w_view = w.view();
  range_policy policy(0, n);

  if (alpha == 1.0) {
    Kokkos::parallel_for(
        "waxpby_alpha", policy, KOKKOS_LAMBDA(const index_type& i) {
          w_view[i] = x_view[i] + beta * y_view[i];
        });
  } else if (beta == 1.0) {
    Kokkos::parallel_for(
        "waxpby_beta", policy, KOKKOS_LAMBDA(const index_type& i) {
          w_view[i] = alpha * x_view[i] + y_view[i];
        });
  } else {
    Kokkos::parallel_for(
        "waxpby", policy, KOKKOS_LAMBDA(const index_type& i) {
          w_view[i] = alpha * x_view[i] + beta * y_view[i];
        });
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KOKKOS_WAXPBY_IMPL_HPP