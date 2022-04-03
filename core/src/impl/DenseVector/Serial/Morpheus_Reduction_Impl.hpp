/**
 * Morpheus_Scan_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_SERIAL_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_SERIAL_REDUCTION_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

// Compute sum reduction using Kahan summation for an accurate sum of large
// arrays. http://en.wikipedia.org/wiki/Kahan_summation_algorithm
template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::index_type size, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using value_type = typename Vector::value_type;
  using index_type = typename Vector::index_type;
  value_type sum   = in[0];
  value_type c     = (value_type)0.0;

  for (index_type i = 1; i < size; i++) {
    value_type y = in[i] - c;
    value_type t = sum + y;
    c            = (t - sum) - y;
    sum          = t;
  }

  return sum;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_SERIAL_REDUCTION_IMPL_HPP