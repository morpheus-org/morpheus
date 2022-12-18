/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#if defined(MORPHEUS_ENABLE_HIP)
#include <impl/Morpheus_HIPUtils.hpp>

#define SHFL_DOWN(mask, val, offset) __shfl_down(val, offset)
#define BALLOT(mask, predicate) __ballot(predicate)
#define BIT_MASK 0xffffffffffffffff
#elif defined(MORPHEUS_ENABLE_CUDA)
#include <impl/Morpheus_CudaUtils.hpp>

#define SHFL_DOWN(mask, val, offset) __shfl_down_sync(mask, val, offset)
#define BALLOT(mask, predicate) __ballot_sync(mask, predicate)
#define BIT_MASK 0xffffffff
#endif

namespace Morpheus {
namespace Impl {
namespace Kernels {

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // mySum += __shfl_down_sync(mask, mySum, offset);
    mySum += SHFL_DOWN(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif

template <typename SizeType, typename ValueType, unsigned int blockSize,
          bool nIsPow2>
__global__ void reduce_kernel(const ValueType *__restrict__ g_idata,
                              ValueType *__restrict__ g_odata, SizeType n) {
  ValueType *sdata = SharedMemory<ValueType>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  SizeType tid      = threadIdx.x;
  SizeType gridSize = blockSize * gridDim.x;
  // unsigned int maskLength = (blockSize & 31);  // 31 = WARP_SIZE-1
  // maskLength              = (maskLength > 0) ? (32 - maskLength) :
  // maskLength;
  SizeType maskLength = (blockSize & (WARP_SIZE - 1));  // 31 = WARP_SIZE-1
  maskLength = (maskLength > 0) ? (WARP_SIZE - maskLength) : maskLength;
  const SizeType mask = (BIT_MASK) >> maskLength;

  ValueType mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    SizeType i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize   = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    SizeType i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<ValueType>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % WARP_SIZE) == 0) {
    sdata[tid / WARP_SIZE] = mySum;
  }

  __syncthreads();

  const SizeType shmem_extent =
      (blockSize / WARP_SIZE) > 0 ? (blockSize / WARP_SIZE) : 1;
  const SizeType ballot_result = BALLOT(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<ValueType>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // _REDUCE_KERNEL_H_