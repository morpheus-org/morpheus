/**
 * Morpheus_Reduction_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KOKKOS_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KOKKOS_REDUCTION_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::size_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using size_type       = typename Vector::size_type;
  using IndexType       = Kokkos::IndexType<size_type>;
  using range_policy    = Kokkos::RangePolicy<IndexType, execution_space>;
  using ValueArray      = typename Vector::value_array_type;
  using value_type      = typename Vector::non_const_value_type;

  const ValueArray in_view = in.const_view();
  range_policy policy(0, size);

  value_type result = value_type(0);
  Kokkos::parallel_reduce(
      "reduce", policy,
      KOKKOS_LAMBDA(const size_type& i, value_type& lsum) {
        lsum += in_view[i];
      },
      result);

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KOKKOS_REDUCTION_IMPL_HPP