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
ValueType reduce(
    const Vector& in, typename Vector::index_type size, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Vector>>* =
        nullptr) {
  // TODO: Invoke reduce function until it produces a single result
  // TODO: Set a threshold for when the final reduction is offloaded to CPU
}

template <class T>
void reduce(int size, int threads, int blocks, T* d_idata, T* d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // For reduce kernel we require only blockSize/warpSize
  // number of elements in shared memory
  smemSize = ((threads / 32) + 1) * sizeof(T);
  if (isPow2(size)) {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<T, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<T, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<T, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<T, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<T, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<T, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<T, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<T, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<T, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<T, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<T, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  } else {
    switch (threads) {
      case 1024:
        Kernels::reduce_kernel<T, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        Kernels::reduce_kernel<T, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        Kernels::reduce_kernel<T, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        Kernels::reduce_kernel<T, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        Kernels::reduce_kernel<T, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        Kernels::reduce_kernel<T, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        Kernels::reduce_kernel<T, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        Kernels::reduce_kernel<T, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        Kernels::reduce_kernel<T, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        Kernels::reduce_kernel<T, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        Kernels::reduce_kernel<T, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_REDUCTION_IMPL_HPP