/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_HIP_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_HIP_DOT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Reduction.hpp>

#include <impl/DenseVector/HIP/Morpheus_Workspace.hpp>

#ifdef MORPHEUS_ENABLE_TPL_HIPBLAS
#include <hipblas.h>
#else
#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Dot_Impl.hpp>
#endif  // MORPHEUS_ENABLE_TPL_HIPBLAS

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2>
typename Vector1::value_type dot(
    const typename Vector1::index_type n, const Vector1& x, const Vector2& y,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using index_type = typename Vector1::index_type;
  using value_type = typename Vector1::non_const_value_type;

  value_type local_result;

#ifdef MORPHEUS_ENABLE_TPL_HIPBLAS
  hipblasdotspace.init();
  hipblasdotspace.allocate<value_type>(1);
  index_type incx = 1, incy = 1;
  hipblasDdot(hipblasdotspace.handle(), n, x.data(), incx, y.data(), incy,
              hipblasdotspace.data<value_type>());

  hipMemcpy(&local_result, hipblasdotspace.data<value_type>(),
            sizeof(value_type), hipMemcpyDeviceToHost);
#else
  hipdotspace.allocate<value_type>(n);

  Kernels::dot_kernel_part1<256, value_type, index_type>
      <<<256, 256>>>(n, x.data(), y.data(), hipdotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("dot: Kernel execution failed");
#endif

  Kernels::dot_kernel_part2<256, value_type>
      <<<1, 256>>>(hipdotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("dot: Kernel execution failed");
#endif

  hipMemcpy(&local_result, hipdotspace.data<value_type>(), sizeof(value_type),
            hipMemcpyDeviceToHost);
#endif  // MORPHEUS_ENABLE_TPL_HIPBLAS

  return local_result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DENSEVECTOR_HIP_DOT_IMPL_HPP
