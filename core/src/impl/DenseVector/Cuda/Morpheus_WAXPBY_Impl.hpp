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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_WAXPBY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_WAXPBY_Impl.hpp>

#include <cassert>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2,
          typename Vector3>
inline void waxpby(
    const size_t n, const typename Vector1::value_type alpha, const Vector1& x,
    const typename Vector2::value_type beta, const Vector2& y, Vector3& w,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::is_dense_vector_format_container_v<Vector3> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2, Vector3>>* =
        nullptr) {
  assert(x.size() >= n);
  assert(y.size() >= n);
  assert(w.size() >= n);

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Kernels::waxpby_kernel<typename Vector1::value_type,
                         typename Vector2::value_type, size_t>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(n, alpha, x.data(), beta, y.data(),
                                      w.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("spmv_waxpby_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_ELEMENTWISE_IMPL_HPP