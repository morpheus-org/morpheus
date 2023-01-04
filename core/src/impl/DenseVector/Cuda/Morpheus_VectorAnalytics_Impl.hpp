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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_VECTORANALYTICS_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_VECTORANALYTICS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/DenseVector/Kokkos/Morpheus_VectorAnalytics_Impl.hpp>
#include <impl/DenseVector/Kernels/Morpheus_VectorAnalytics_Impl.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Segmented_Reduction_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
typename Vector::value_type max(
    const Vector& in, typename Vector::size_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
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
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
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
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
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
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, VectorIn, VectorOut>>* = nullptr) {
  using size_type  = typename VectorIn::size_type;
  using index_type = typename VectorIn::value_type;
  using value_type = typename VectorOut::value_type;

  Kokkos::sort(in.const_view());

  VectorOut vals(in.size(), 1);

  if (in.size() == 0) {
    // empty vector
    return;
  } else if (in.size() < static_cast<size_type>(WARP_SIZE)) {
    // small vector
    Kernels::count_occurences_dense_vector_serial_kernel<size_type, index_type,
                                                         value_type>
        <<<1, 1, 0>>>(in.size(), in.data(), vals.data(), out.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
    getLastCudaError(
        "count_occurences_dense_vector_serial_kernel: Kernel execution failed");
#endif
    return;
  }

  const size_type BLOCK_SIZE = 256;
  const size_type MAX_BLOCKS =
      max_active_blocks(Kernels::count_occurences_dense_vector_flat_kernel<
                            size_type, index_type, value_type, BLOCK_SIZE>,
                        BLOCK_SIZE, 0);
  const size_type WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

  const size_type num_units = in.size() / WARP_SIZE;
  const size_type num_warps = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
  const size_type num_blocks =
      Impl::ceil_div<size_type>(num_warps, WARPS_PER_BLOCK);
  const size_type num_iters = Impl::ceil_div<size_type>(num_units, num_warps);

  const size_type interval_size = WARP_SIZE * num_iters;
  // do the last few nonzeros separately (fewer than WARP_SIZE elements)
  const size_type tail = num_units * WARP_SIZE;

  const size_type active_warps =
      (interval_size == 0) ? 0 : Impl::ceil_div<size_type>(tail, interval_size);

  VectorIn temp_keys(active_warps, 0);
  VectorOut temp_vals(active_warps, 0);

  Kernels::count_occurences_dense_vector_flat_kernel<size_type, index_type,
                                                     value_type, BLOCK_SIZE>
      <<<num_blocks, BLOCK_SIZE, 0>>>(tail, interval_size, in.data(),
                                      vals.data(), out.data(), temp_keys.data(),
                                      temp_vals.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError(
      "count_occurences_dense_vector_flat_kernel: Kernel execution failed");
#endif

  Kernels::reduce_update_kernel<size_type, index_type, value_type, BLOCK_SIZE>
      <<<1, BLOCK_SIZE, 0>>>(active_warps, temp_keys.data(), temp_vals.data(),
                             out.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("reduce_update_kernel: Kernel execution failed");
#endif

  Kernels::count_occurences_dense_vector_serial_kernel<size_type, index_type,
                                                       value_type><<<1, 1, 0>>>(
      in.size() - tail, in.data() + tail, vals.data() + tail, out.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError(
      "count_occurences_dense_vector_serial_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_VECTORANALYTICS_IMPL_HPP