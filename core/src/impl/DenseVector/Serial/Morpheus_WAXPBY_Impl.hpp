/**
 * Morpheus_WAXPBY_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_SERIAL_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_SERIAL_WAXPBY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {
template <typename ExecSpace, typename Vector>
inline void waxpby(
    const typename Vector::index_type n,
    const typename Vector::value_type alpha, const Vector& x,
    const typename Vector::value_type beta, const Vector& y, Vector& w,
    DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using index_type = typename Vector::index_type;

  if (alpha == 1.0) {
    for (index_type i = 0; i < n; i++) w[i] = x[i] + beta * y[i];
  } else if (beta == 1.0) {
    for (index_type i = 0; i < n; i++) w[i] = alpha * x[i] + y[i];
  } else {
    for (index_type i = 0; i < n; i++) w[i] = alpha * x[i] + beta * y[i];
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_SERIAL_WAXPBY_IMPL_HPP