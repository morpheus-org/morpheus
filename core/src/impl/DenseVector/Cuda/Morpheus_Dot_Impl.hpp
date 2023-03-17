/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_Reduction.hpp>

#include <impl/DenseVector/Cuda/Morpheus_Workspace.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Dot_Impl.hpp>

#ifdef MORPHEUS_ENABLE_TPL_CUBLAS
#include <Morpheus_TypeTraits.hpp>
#include <cublas_v2.h>
#endif  // MORPHEUS_ENABLE_TPL_CUBLAS

namespace Morpheus {
namespace Impl {

template <typename Vector1, typename Vector2>
typename Vector2::value_type dot_ref(const typename Vector1::size_type n,
                                     const Vector1& x, const Vector2& y);
template <typename SizeType>
double dot_cublas(const SizeType n, const double* x, int incx, const double* y,
                  int incy);
template <typename SizeType>
double dot_cublas(const SizeType n, const float* x, int incx, const float* y,
                  int incy);

template <typename ExecSpace, typename Vector1, typename Vector2>
typename Vector2::value_type dot(
    const typename Vector1::size_type n, const Vector1& x, const Vector2& y,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using value_type1 = typename Vector1::non_const_value_type;
  using value_type2 = typename Vector2::value_type;

  value_type2 local_result;

#ifdef MORPHEUS_ENABLE_TPL_CUBLAS
  using index_type = typename Vector1::index_type;
  using val_t =
      typename std::remove_pointer_t<Morpheus::remove_cvref_t<value_type1>>;
  if constexpr (std::is_floating_point_v<val_t>) {
    index_type incx = 1, incy = 1;
    local_result = dot_cublas(n, x.data(), incx, y.data(), incy);
  } else {
    local_result = dot_ref(n, x, y);
  }
#else
  local_result = dot_ref(n, x, y);
#endif  // MORPHEUS_ENABLE_TPL_CUBLAS

  return local_result;
}

template <typename SizeType>
double dot_cublas(const SizeType n, const double* x, int incx, const double* y,
                  int incy) {
  double lres = 0;
  cublasdotspace.init();
  cublasdotspace.allocate<double>(1);
  cublasDdot(cublasdotspace.handle(), n, x, incx, y, incy,
             (double*)cublasdotspace.data<double>());

  checkCudaErrors(cudaMemcpy(&lres, cublasdotspace.data<double>(),
                             sizeof(double), cudaMemcpyDeviceToHost));

  return lres;
}

template <typename SizeType>
float dot_cublas(const SizeType n, const float* x, int incx, const float* y,
                 int incy) {
  float lres = 0;
  cublasdotspace.init();
  cublasdotspace.allocate<float>(1);
  cublasDdot(cublasdotspace.handle(), n, x, incx, y, incy,
             (float*)cublasdotspace.data<float>());

  checkCudaErrors(cudaMemcpy(&lres, cublasdotspace.data<float>(), sizeof(float),
                             cudaMemcpyDeviceToHost));

  return lres;
}

template <typename Vector1, typename Vector2>
typename Vector2::value_type dot_ref(const typename Vector1::size_type n,
                                     const Vector1& x, const Vector2& y) {
  using size_type  = typename Vector1::size_type;
  using value_type = typename Vector2::value_type;

  value_type lres = 0;
  cudotspace.allocate<value_type>(n);

  Kernels::dot_kernel_part1<256, value_type, size_type>
      <<<256, 256>>>(n, x.data(), y.data(), cudotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("dot: Kernel execution failed");
#endif

  Kernels::dot_kernel_part2<256, value_type>
      <<<1, 256>>>(cudotspace.data<value_type>());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("dot: Kernel execution failed");
#endif

  cudaMemcpy(&lres, cudotspace.data<value_type>(), sizeof(value_type),
             cudaMemcpyDeviceToHost);
  return lres;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP
