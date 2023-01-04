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

#ifndef MORPHEUS_DENSEVECTOR_KOKKOS_VECTORANALYTICS_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KOKKOS_VECTORANALYTICS_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type max(
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

  value_type result;
  Kokkos::parallel_reduce(
      "max", policy,
      KOKKOS_LAMBDA(const size_type& i, value_type& lmax) {
        lmax = lmax > in_view[i] ? lmax : in_view[i];
      },
      Kokkos::Max<value_type>(result));
  Kokkos::fence();

  return result;
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type min(
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

  value_type result;
  Kokkos::parallel_reduce(
      "min", policy,
      KOKKOS_LAMBDA(const size_type& i, value_type& lmin) {
        lmin = lmin < in_view[i] ? lmin : in_view[i];
      },
      Kokkos::Min<value_type>(result));
  Kokkos::fence();

  return result;
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type std(
    const Vector& in, typename Vector::size_type size,
    typename Vector::value_type mean,
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
      "squared_sum", policy,
      KOKKOS_LAMBDA(const size_type& i, value_type& lsum) {
        lsum += (in_view[i] - mean) * (in_view[i] - mean);
      },
      result);
  Kokkos::fence();

  return sqrt(result / (value_type)size);
}

template <typename ExecSpace, typename VectorIn, typename VectorOut>
void count_occurences(
    const VectorIn& in, VectorOut& out,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<VectorIn> &&
        Morpheus::is_dense_vector_format_container_v<VectorOut> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, VectorIn, VectorOut>>* = nullptr) {
  using backend = Morpheus::CustomBackend<typename ExecSpace::execution_space>;
  Impl::count_occurences<backend>(in, out);
}

template <typename ExecSpace, typename Vector>
typename Vector::size_type count_nnz(
    const Vector& vec, typename Vector::value_type threshold,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using size_type       = typename Vector::size_type;
  using IndexType       = Kokkos::IndexType<size_type>;
  using range_policy    = Kokkos::RangePolicy<IndexType, execution_space>;
  using ValueArray      = typename Vector::value_array_type;

  const ValueArray in_view = vec.const_view();
  range_policy policy(0, vec.size());

  size_type result = 0;
  Kokkos::parallel_reduce(
      "count_nnz", policy,
      KOKKOS_LAMBDA(const size_type& i, size_type& lsum) {
        if (in_view[i] > threshold) {
          lsum += 1;
        }
      },
      result);
  Kokkos::fence();

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KOKKOS_VECTORANALYTICS_IMPL_HPP