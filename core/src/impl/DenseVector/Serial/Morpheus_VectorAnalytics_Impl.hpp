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

#ifndef MORPHEUS_DENSEVECTOR_SERIAL_VECTORANALYTICS_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_SERIAL_VECTORANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/DenseVector/Kokkos/Morpheus_VectorAnalytics_Impl.hpp>
#include <impl/DenseVector/Serial/Morpheus_Scan_Impl.hpp>

#include <Kokkos_Sort.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type max(
    const Vector& in, typename Vector::size_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
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
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::min<backend>(in, size);
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type std(
    const Vector& in, typename Vector::size_type size,
    typename Vector::value_type mean,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using backend = Morpheus::GenericBackend<typename ExecSpace::execution_space>;
  return Impl::std<backend>(in, size, mean);
}

template <typename ExecSpace, typename VectorIn, typename VectorOut>
void count_occurences(
    const VectorIn& in, VectorOut& out,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<VectorIn> &&
        Morpheus::is_dense_vector_format_container_v<VectorOut> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, VectorIn, VectorOut>>* = nullptr) {
  using size_type  = typename VectorIn::size_type;
  using value_type = typename VectorOut::value_type;
  using index_type = typename VectorIn::value_type;

  Kokkos::sort(in.const_view());

  VectorOut vals(in.size(), 1);
  index_type prev_key   = in[0];
  value_type prev_value = vals[0];

  out[in[0]] = vals[0];
  for (size_type i = 1; i < in.size(); i++) {
    index_type key = in[i];
    if (prev_key == key) {
      out[key] = prev_value = prev_value + vals[i];
    } else {
      out[key] = prev_value = vals[i];
    }

    prev_key = key;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DENSEVECTOR_SERIAL_VECTORANALYTICS_IMPL_HPP