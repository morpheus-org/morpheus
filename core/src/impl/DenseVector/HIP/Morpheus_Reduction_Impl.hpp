/**
 * Morpheus_Reduction_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_HIP_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_HIP_REDUCTION_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_Copy.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>

#include <impl/DenseVector/Kernels/Morpheus_Reduction_Impl.hpp>
#include <impl/DenseVector/Serial/Morpheus_Reduction_Impl.hpp>
#include <impl/Morpheus_HIPUtils.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
void reduce(
    const Vector& in, Vector& out, unsigned int size, int threads, int blocks,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using value_type = typename Vector::value_type;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  value_type* d_idata = in.data();
  value_type* d_odata = out.data();

  // For reduce kernel we require only blockSize/warpSize
  // number of elements in shared memory
  int smemSize = ((threads / 32) + 1) * sizeof(value_type);
  if (isPow2<unsigned int>(size)) {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<value_type, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<value_type, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<value_type, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<value_type, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<value_type, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<value_type, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<value_type, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<value_type, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<value_type, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<value_type, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<value_type, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<value_type, 1024, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<value_type, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<value_type, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<value_type, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<value_type, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<value_type, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<value_type, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<value_type, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<value_type, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<value_type, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<value_type, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::index_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using value_type = typename Vector::value_type;

  value_type result           = 0;
  const int maxThreads        = 256;  // number of threads per block
  const int maxBlocks         = min(size / maxThreads, MAX_BLOCK_DIM_SIZE);
  const int cpuFinalThreshold = WARP_SIZE;
  int numBlocks               = 0;
  int numThreads              = 0;

  getNumBlocksAndThreads<int>(size, maxBlocks, maxThreads, numBlocks,
                              numThreads);

  Vector inter_sums(maxBlocks, 0);
  Vector out(numBlocks, 0);
  Impl::reduce<ExecSpace>(in, out, size, numThreads, numBlocks);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastHIPError("reduce_kernel: Kernel execution failed");
#endif

  // sum partial block sums on GPU
  int s = numBlocks;

  while (s > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads<int>(s, maxBlocks, maxThreads, blocks, threads);
    Morpheus::copy(out, inter_sums, 0, s);

    Impl::reduce<ExecSpace>(inter_sums, out, s, threads, blocks);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
    getLastHIPError("reduce_kernel: Kernel execution failed");
#endif

    s = (s + (threads * 2 - 1)) / (threads * 2);
  }

  if (s > 1) {
    typename Vector::HostMirror h_out(s, 0);
    // copy result from device to host
    Morpheus::copy(out, h_out, 0, s);
    result = reduce<Kokkos::Serial>(h_out, s);
  } else {
    // copy final sum from device to host
    typename Vector::HostMirror h_out(1, 0);
    Morpheus::copy(out, h_out, 0, 1);
    result = h_out[0];
  }

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DENSEVECTOR_HIP_REDUCTION_IMPL_HPP