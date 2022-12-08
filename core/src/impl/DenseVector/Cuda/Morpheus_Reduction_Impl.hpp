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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/DenseVector/Morpheus_Copy_Impl.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Reduction_Impl.hpp>
#include <impl/DenseVector/Serial/Morpheus_Reduction_Impl.hpp>
#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Morpheus_Copy_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
void reduce(const Vector& in, Vector& out, unsigned int size, int threads,
            int blocks,
            typename std::enable_if_t<
                Morpheus::is_dense_vector_format_container_v<Vector> &&
                Morpheus::has_custom_backend_v<ExecSpace> &&
                Morpheus::has_cuda_execution_space_v<ExecSpace> &&
                Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using value_type = typename Vector::value_type;

  value_type* d_idata = in.data();
  value_type* d_odata = out.data();

  // For reduce kernel we require only blockSize/warpSize
  // number of elements in shared memory
  int smemSize = ((threads / 32) + 1) * sizeof(value_type);
  if (isPow2<unsigned int>(size)) {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<value_type, 1024, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<value_type, 512, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<value_type, 256, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<value_type, 128, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<value_type, 64, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<value_type, 32, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<value_type, 16, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<value_type, 8, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<value_type, 4, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<value_type, 2, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<value_type, 1, true>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<value_type, 1024, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<value_type, 512, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<value_type, 256, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<value_type, 128, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<value_type, 64, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<value_type, 32, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<value_type, 16, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<value_type, 8, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<value_type, 4, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<value_type, 2, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<value_type, 1, false>
            <<<blocks, threads, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::index_type size,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using value_type = typename Vector::value_type;

  value_type result    = 0;
  const int maxThreads = 256;  // number of threads per block
  const int maxBlocks =
      min((size_t)size / maxThreads + 1, (size_t)MAX_BLOCK_DIM_SIZE);
  const int cpuFinalThreshold = WARP_SIZE;
  int numBlocks               = 0;
  int numThreads              = 0;

  getNumBlocksAndThreads<int>(size, maxBlocks, maxThreads, numBlocks,
                              numThreads);

  Vector inter_sums(maxBlocks, 0);
  Vector out(numBlocks, 0);
  reduce<ExecSpace>(in, out, size, numThreads, numBlocks);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("reduce_kernel: Kernel execution failed");
#endif

  // sum partial block sums on GPU
  int reduced_size = numBlocks;

  while (reduced_size > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads<int>(reduced_size, maxBlocks, maxThreads, blocks,
                                threads);
    Impl::copy(out, inter_sums, 0, reduced_size, 0, reduced_size);

    Impl::reduce<ExecSpace>(inter_sums, out, reduced_size, threads, blocks);
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
    getLastCudaError("reduce_kernel: Kernel execution failed");
#endif

    reduced_size = (reduced_size + (threads * 2 - 1)) / (threads * 2);
  }

  if (reduced_size > 1) {
    typename Vector::HostMirror h_out(reduced_size, 0);
    // copy result from device to host
    Impl::copy(out, h_out, 0, reduced_size, 0, reduced_size);
    result = Impl::reduce<Morpheus::Serial>(h_out, reduced_size);
  } else {
    // copy final sum from device to host
    typename Vector::HostMirror h_out(1, 0);
    Impl::copy(out, h_out, 0, 1, 0, 1);
    result = h_out[0];
  }

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP