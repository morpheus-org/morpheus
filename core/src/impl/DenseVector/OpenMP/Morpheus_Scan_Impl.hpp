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

#ifndef MORPHEUS_DENSEVECTOR_OPENMP_SCAN_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_OPENMP_SCAN_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
void inclusive_scan(
    const Vector& in, Vector& out, typename Vector::index_type size,
    typename Vector::index_type start, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

#if _OPENMP >= 201511
  value_type initial = 0;
// TODO: Scan semantics require OpenMP5
#pragma omp simd reduction(inscan, + : initial)
  for (index_type i = start; i < size; i++) {
    initial += in[i];
#pragma omp scan inclusive(initial)
    out[i] = initial;
  }
#else
  static_assert("Requires OpenMP4.5 and above");
#endif
}

template <typename ExecSpace, typename Vector>
void exclusive_scan(
    const Vector& in, Vector& out, typename Vector::index_type size,
    typename Vector::index_type start, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

#if _OPENMP >= 201511
  if (size > 0) {
    value_type initial = 0;
    // TODO: Scan semantics require OpenMP5
#pragma omp simd reduction(inscan, + : initial)
    for (index_type i = start; i < size - 1; i++) {
      out[i] = initial;
#pragma omp scan exclusive(initial)
      initial += in[i];
    }
    out[size - 1] = initial;
  }
#else
  static_assert("Requires OpenMP4.5 and above");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DENSEVECTOR_OPENMP_SCAN_IMPL_HPP