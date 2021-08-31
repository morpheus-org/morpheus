/**
 * Morpheus_CudaUtils.hpp
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

#ifndef MORPHEUS_CUDA_UTILS_HPP
#define MORPHEUS_CUDA_UTILS_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <cuda.h>

namespace Morpheus {
namespace Impl {

// maximum number of co-resident threads
const int CUDA_MAX_THREADS = (30 * 1024);
const int CUDA_WARP_SIZE   = 32;

template <typename Size1, typename Size2>
__host__ __device__ Size1 DIVIDE_INTO(Size1 N, Size2 granularity) {
  return (N + (granularity - 1)) / granularity;
}

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE,
                         const size_t dynamic_smem_bytes) {
  int MAX_BLOCKS;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &MAX_BLOCKS, kernel, (int)CTA_SIZE, dynamic_smem_bytes);
  return (size_t)MAX_BLOCKS;
}

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel. We set threads / block to the minimum of maxThreads and n/2.
// We observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
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

  blocks = MIN(maxBlocks, blocks);
}

template <typename T>
__device__ T min(T x, T y) {
  return x < y ? x : y;
}

template <typename T>
__device__ T max(T x, T y) {
  return x > y ? y : x;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_CUDA_UTILS_HPP