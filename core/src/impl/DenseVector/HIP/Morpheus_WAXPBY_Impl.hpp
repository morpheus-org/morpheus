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

#ifndef MORPHEUS_DENSEVECTOR_HIP_WAXPBY_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_HIP_WAXPBY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_WAXPBY_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
inline void waxpby(
    const typename Vector::index_type n,
    const typename Vector::value_type alpha, const Vector& x,
    const typename Vector::value_type beta, const Vector& y, Vector& w,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Kernels::waxpby_kernel<value_type, index_type><<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(
      n, alpha, x.data(), beta, y.data(), w.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("spmv_waxpby_kernel: Kernel execution failed");
#endif
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DENSEVECTOR_HIP_WAXPBY_IMPL_HPP