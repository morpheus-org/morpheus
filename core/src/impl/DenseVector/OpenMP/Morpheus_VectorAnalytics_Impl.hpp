/**
 * Morpheus_VectorAnalytics_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_OPENMP_VECTORANALYTICS_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_OPENMP_VECTORANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/DenseVector/Kokkos/Morpheus_VectorAnalytics_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type max(
    const Vector& in, typename Vector::size_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::max<backend>(in, size);
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type min(
    const Vector& in, typename Vector::size_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::min<backend>(in, size);
}

template <typename ExecSpace, typename Vector>
double std(const Vector& in, typename Vector::size_type size,
           typename Vector::value_type mean,
           typename std::enable_if_t<
               Morpheus::is_dense_vector_format_container_v<Vector> &&
               Morpheus::has_custom_backend_v<ExecSpace> &&
               Morpheus::has_openmp_execution_space_v<ExecSpace> &&
               Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::std<backend>(in, size, mean);
}

template <typename ExecSpace, typename Vector>
typename Vector::size_type count_nnz(
    const Vector& vec, typename Vector::value_type threshold,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::count_nnz<backend>(vec, threshold);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DENSEVECTOR_OPENMP_VECTORANALYTICS_IMPL_HPP