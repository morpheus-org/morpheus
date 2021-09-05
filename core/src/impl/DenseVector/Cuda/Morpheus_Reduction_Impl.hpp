/**
 * Morpheus_Reduction_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/DenseVector/Kernels/Morpheus_Reduction_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
void reduce(
    const Vector& in, Vector& out, unsigned int size, int threads, int blocks,
    DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using ValueType = typename Vector::value_type;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  ValueType* d_idata = in.data();
  ValueType* d_odata = out.data();

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(ValueType)
                                 : threads * sizeof(ValueType);

  // For reduce kernel we require only blockSize/warpSize
  // number of elements in shared memory
  smemSize = ((threads / 32) + 1) * sizeof(ValueType);
  if (isPow2<unsigned int>(size)) {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<ValueType, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<ValueType, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<ValueType, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<ValueType, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<ValueType, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<ValueType, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<ValueType, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<ValueType, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<ValueType, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<ValueType, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<ValueType, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<ValueType, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<ValueType, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<ValueType, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<ValueType, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<ValueType, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<ValueType, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<ValueType, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<ValueType, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<ValueType, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<ValueType, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<ValueType, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

template <typename ExecSpace, typename Vector>
typename Vector::value_type reduce(
    const Vector& in, typename Vector::index_type size, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  using ValueType = typename Vector::value_type;

  ValueType result            = 0;
  const int maxThreads        = 256;  // number of threads per block
  const int maxBlocks         = min(size / maxThreads, CUDA_MAX_BLOCK_DIM_SIZE);
  const int cpuFinalThreshold = CUDA_WARP_SIZE;
  int numBlocks               = 0;
  int numThreads              = 0;

  getNumBlocksAndThreads<int>(size, maxBlocks, maxThreads, numBlocks,
                              numThreads);

  Vector inter_sums(maxBlocks, 0);
  Vector out(numBlocks, 0);
  reduce<ExecSpace>(in, out, size, numThreads, numBlocks,
                    typename Vector::tag{}, typename Vector::tag{}, Alg0{});

  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  // sum partial block sums on GPU
  int s = numBlocks;

  while (s > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads<int>(s, maxBlocks, maxThreads, blocks, threads);
    checkCudaErrors(cudaMemcpy(inter_sums.data(), out.data(),
                               s * sizeof(ValueType),
                               cudaMemcpyDeviceToDevice));
    reduce<ExecSpace>(inter_sums, out, s, threads, blocks,
                      typename Vector::tag{}, typename Vector::tag{}, Alg0{});

    s = (s + (threads * 2 - 1)) / (threads * 2);
  }

  if (s > 1) {
    typename Vector::HostMirror h_out(s, 0);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_out.data(), out.data(), s * sizeof(ValueType),
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < s; i++) {
      result += h_out[i];
    }

  } else {
    // copy final sum from device to host
    checkCudaErrors(cudaMemcpy(&result, out.data(), sizeof(ValueType),
                               cudaMemcpyDeviceToHost));
  }

  return result;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP