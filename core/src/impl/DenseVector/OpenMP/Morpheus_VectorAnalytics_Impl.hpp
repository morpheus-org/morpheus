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

#include <Kokkos_Sort.hpp>

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
typename Vector::value_type std(
    const Vector& in, typename Vector::size_type size,
    typename Vector::value_type mean,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
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
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, VectorIn, VectorOut>>* = nullptr) {
  using size_type  = typename VectorIn::size_type;
  using value_type = typename VectorOut::value_type;
  using index_type = typename VectorIn::value_type;

  Kokkos::sort(in.const_view());

  VectorOut vals(in.size(), 1);

  const size_type keys         = in.size();
  const size_type sentinel_key = keys + 1;

#pragma omp parallel
  {
    const size_type num_threads = omp_get_num_threads();
    const size_type work_per_thread =
        Impl::ceil_div<size_type>(keys, num_threads);
    const size_type thread_id = omp_get_thread_num();
    const size_type begin     = work_per_thread * thread_id;
    const size_type end       = std::min(begin + work_per_thread, keys);
    size_type n               = begin;

    if (begin < end) {
      const index_type first = begin > 0 ? in[begin - 1] : sentinel_key;
      const index_type last  = end < keys ? in[end] : sentinel_key;

      // handle key overlap with previous thread
      if (first != (index_type)sentinel_key) {
        value_type partial_sum = value_type(0);
        for (; n < end && in[n] == first; n++) {
          partial_sum += vals[n];
        }
        Impl::atomic_add(&out[first], partial_sum);
      }

      // handle non-overlapping keys
      for (; n < end && in[n] != last; n++) {
        out[in[n]] += vals[n];
      }

      // handle key overlap with following thread
      if (last != (index_type)sentinel_key) {
        value_type partial_sum = value_type(0);
        for (; n < end; n++) {
          partial_sum += vals[n];
        }
        Impl::atomic_add(&out[last], partial_sum);
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DENSEVECTOR_OPENMP_VECTORANALYTICS_IMPL_HPP