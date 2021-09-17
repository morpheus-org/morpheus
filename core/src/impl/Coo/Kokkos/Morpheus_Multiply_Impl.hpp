/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {
template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, CooTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;

  const size_t BLOCK_SIZE = 256;
#if defined(MORPHEUS_ENABLE_CUDA)
  const size_t WARP_SZ =
      std::is_same<execution_space, Kokkos::Cuda>::value ? 32 : 1;
#else
  const size_t WARP_SZ = 1;  // Can that be vector sz?
#endif
  const size_t MAX_BLOCKS      = execution_space().concurency();
  const size_t WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SZ;

  const size_t num_units  = A.nnnz() / WARP_SZ;
  const size_t num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
  const size_t num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
  const size_t num_iters  = DIVIDE_INTO(num_units, num_warps);

  const IndexType interval_size = WARP_SZ * num_iters;

  const IndexType tail =
      num_units * WARP_SZ;  // do the last few nonzeros separately (fewer
                            // than WARP_SIZE elements)

  const unsigned int active_warps =
      (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);

  std::cout << "MAX_BLOCKS(" << MAX_BLOCKS << ")\t"
            << "WARPS_PER_BLOCK(" << WARPS_PER_BLOCK << ")\t"
            << "num_units(" << num_units << ")\t"
            << "num_warps(" << num_warps << ")\t"
            << "num_blocks(" << num_blocks << ")\t"
            << "num_iters(" << num_iters << ")\t"
            << "interval_size(" << interval_size << ")\t"
            << "tail(" << tail << ")\t"
            << "active_warps(" << active_warps << ")\t" << std::endl;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP