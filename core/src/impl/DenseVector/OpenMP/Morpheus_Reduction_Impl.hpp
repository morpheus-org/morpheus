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

#ifndef MORPHEUS_DENSEVECTOR_OPENMP_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_OPENMP_REDUCTION_IMPL_HPP

#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::index_type size, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using value_type = typename Vector::value_type;
  using index_type = typename Vector::index_type;

  value_type sum = value_type(0);

#pragma omp parallel for reduction(+ : sum)
  for (index_type i = 0; i < size; i++) sum += in[i];

  return sum;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP

#endif  // MORPHEUS_DENSEVECTOR_OPENMP_REDUCTION_IMPL_HPP
