/**
 * Morpheus_Dot_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_KERNELS_DOT_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_DOT_IMPL_HPP

#if defined(MORPHEUS_ENABLE_HIP)
#include <impl/Morpheus_HIPUtils.hpp>
#elif defined(MORPHEUS_ENABLE_CUDA)
#include <impl/Morpheus_CudaUtils.hpp>
#endif

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename ValueType, typename SizeType>
__global__ void dot_kernel(SizeType n, const ValueType* x, const ValueType* y,
                           SizeType* res) {
  const SizeType tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > n) return;

  res[tid] = x[tid] * y[tid];
}

template <unsigned int BLOCKSIZE, typename ValueType, typename SizeType>
__launch_bounds__(BLOCKSIZE) __global__
    void dot_kernel_part1(SizeType n, const ValueType* x, const ValueType* y,
                          ValueType* workspace) {
  SizeType gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
  SizeType inc = gridDim.x * BLOCKSIZE;

  ValueType sum = 0.0;
  for (SizeType idx = gid; idx < n; idx += inc) {
    sum += y[idx] * x[idx];
  }

  __shared__ ValueType sdata[BLOCKSIZE];
  sdata[threadIdx.x] = sum;

  __syncthreads();

  if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) sdata[threadIdx.x] += sdata[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) sdata[threadIdx.x] += sdata[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) sdata[threadIdx.x] += sdata[threadIdx.x + 2];
  __syncthreads();

  if (threadIdx.x == 0) {
    workspace[blockIdx.x] = sdata[0] + sdata[1];
  }
}

template <unsigned int BLOCKSIZE, typename ValueType>
__launch_bounds__(BLOCKSIZE) __global__
    void dot_kernel_part2(ValueType* workspace) {
  __shared__ ValueType sdata[BLOCKSIZE];
  sdata[threadIdx.x] = workspace[threadIdx.x];

  __syncthreads();

  if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8) sdata[threadIdx.x] += sdata[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4) sdata[threadIdx.x] += sdata[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2) sdata[threadIdx.x] += sdata[threadIdx.x + 2];
  __syncthreads();

  if (threadIdx.x == 0) {
    workspace[0] = sdata[0] + sdata[1];
  }
}

template <typename ValueType, typename SizeType>
__global__ void DOT_D_ini(SizeType n, ValueType* x, ValueType* y,
                          ValueType* valpha) {
  extern __shared__ ValueType vtmp[];
  // Each thread loads two elements from each chunk
  // from global to shared memory
  SizeType tid              = threadIdx.x;
  SizeType NumBlk           = gridDim.x;   // = 256
  SizeType BlkSize          = blockDim.x;  // = 192
  SizeType Chunk            = 2 * NumBlk * BlkSize;
  SizeType i                = blockIdx.x * (2 * BlkSize) + tid;
  volatile ValueType* vtmp2 = vtmp;

  // Reduce from n to NumBlk * BlkSize elements. Each thread // operates with
  // two elements of each chunk
  vtmp[tid] = 0;
  while (i < n) {
    vtmp[tid] += x[i] * y[i];
    vtmp[tid] += (i + BlkSize < n) ? (x[i + BlkSize] * y[i + BlkSize]) : 0;
    i += Chunk;
  }
  __syncthreads();
  // Reduce from BlkSize=192 elements to 96, 48, 24, 12, 6, 3 and 1
  if (tid < 96) {
    vtmp[tid] += vtmp[tid + 96];
  }
  __syncthreads();
  if (tid < 48) {
    vtmp[tid] += vtmp[tid + 48];
  }
  __syncthreads();
  if (tid < 24) {
    vtmp2[tid] += vtmp2[tid + 24];
    vtmp2[tid] += vtmp2[tid + 12];
    vtmp2[tid] += vtmp2[tid + 6];
    vtmp2[tid] += vtmp2[tid + 3];
  }
  // Write result for this block to global mem
  if (tid == 0) valpha[blockIdx.x] = vtmp[0] + vtmp[1] + vtmp[2];
}

template <typename ValueType, typename SizeType>
__global__ void DOT_D_fin(ValueType* valpha) {
  extern __shared__ ValueType vtmp[];
  // Each thread loads one element from global to shared mem
  SizeType tid              = threadIdx.x;
  volatile ValueType* vtmp2 = vtmp;
  vtmp[tid]                 = valpha[tid];
  __syncthreads();
  // Reduce from 256 elements to 128, 64, 32, 16, 8, 2 and 1
  if (tid < 128) {
    vtmp[tid] += vtmp[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    vtmp[tid] += vtmp[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    vtmp2[tid] += vtmp2[tid + 32];
    vtmp2[tid] += vtmp2[tid + 16];
    vtmp2[tid] += vtmp2[tid + 8];
    vtmp2[tid] += vtmp2[tid + 4];
    vtmp2[tid] += vtmp2[tid + 2];
    vtmp2[tid] += vtmp2[tid + 1];
  }
  // Write result for this block to global mem
  if (tid == 0) valpha[blockIdx.x] = *vtmp;
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_DOT_IMPL_HPP