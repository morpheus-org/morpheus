/**
 * Morpheus_Dot_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>
#include <Morpheus_Reduction.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Dot_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector1, typename Vector2>
typename Vector1::value_type dot(
    const typename Vector1::index_type n, const Vector1& x, const Vector2& y,
    DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector1,
                               Vector2>>* = nullptr) {
  using index_type = typename Vector1::index_type;
  using value_type = typename Vector1::non_const_value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (x.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Vector1 res_vec(n, 0);

  // execute the dot product kernel
  Kernels::dot_kernel<value_type, index_type><<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(
      x.size(), x.data(), y.data(), res_vec.data());

  return Morpheus::reduce<ExecSpace>(res_vec, n);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_DOT_IMPL_HPP
