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

#ifndef MORPHEUS_DENSEVECTOR_KOKKOS_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KOKKOS_DOT_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2>
inline typename Vector2::value_type dot(
    typename Vector1::index_type n, const Vector1& x, const Vector2& y,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using IndexType       = Kokkos::IndexType<typename Vector1::index_type>;
  using range_policy    = Kokkos::RangePolicy<IndexType, execution_space>;
  using ValueArray      = typename Vector1::value_array_type;
  using value_type      = typename Vector2::non_const_value_type;
  using index_type      = typename Vector1::index_type;

  const ValueArray x_view = x.const_view(), y_view = y.const_view();
  range_policy policy(0, n);

  value_type result = value_type(0);

  if (y.data() == x.data()) {
    Kokkos::parallel_reduce(
        "dot_same_data", policy,
        KOKKOS_LAMBDA(const index_type& i, value_type& lsum) {
          lsum += x_view[i] * x_view[i];
        },
        result);
  } else {
    Kokkos::parallel_reduce(
        "dot", policy,
        KOKKOS_LAMBDA(const index_type& i, value_type& lsum) {
          lsum += x_view[i] * y_view[i];
        },
        result);
  }

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KOKKOS_DOT_IMPL_HPP