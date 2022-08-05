/**
 * Morpheus_CudaUtils.hpp
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

#ifndef MORPHEUS_CUDA_UTILS_HPP
#define MORPHEUS_CUDA_UTILS_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <impl/Morpheus_Utils.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

namespace Morpheus {
namespace Impl {

// maximum number of co-resident threads
const int CUDA_MAX_BLOCK_DIM_SIZE = 65535;
const int CUDA_MAX_THREADS        = (30 * 1024);
const int CUDA_WARP_SIZE          = 32;

template <typename T>
static const char *_cudaGetErrorEnum(T error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE,
                         const size_t dynamic_smem_bytes) {
  int MAX_BLOCKS;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &MAX_BLOCKS, kernel, (int)CTA_SIZE, dynamic_smem_bytes);

  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  return (size_t)MAX_BLOCKS * prop.multiProcessorCount;
}

// Compute the number of threads and blocks to use for the given reduction
// kernel. We set threads / block to the minimum of maxThreads and n/2.
// We observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
template <typename IndexType>
void getNumBlocksAndThreads(
    IndexType n, IndexType maxBlocks, IndexType maxThreads, IndexType &blocks,
    IndexType &threads,
    typename std::enable_if<std::is_integral<IndexType>::value>::type * =
        nullptr) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
  blocks  = (n + (threads * 2 - 1)) / (threads * 2);

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  blocks = min(maxBlocks, blocks);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_CUDA_UTILS_HPP