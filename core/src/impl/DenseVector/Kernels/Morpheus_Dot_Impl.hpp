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

#include <cuda.h>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename ValueType, typename IndexType>
__global__ void dot_kernel(IndexType n, const ValueType* x, const ValueType* y,
                           ValueType* res) {
  const IndexType tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > n) return;

  res[tid] = x[tid] * y[tid];
}

template <unsigned int BLOCKSIZE, typename ValueType, typename IndexType>
__launch_bounds__(BLOCKSIZE)
__global__ void dot_kernel_part1(IndexType n,
                                  const ValueType* x,
                                  const ValueType* y,
                                  ValueType* workspace)
{
    IndexType gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    IndexType inc = gridDim.x * BLOCKSIZE;

    ValueType sum = 0.0;
    for(IndexType idx = gid; idx < n; idx += inc)
    {
        sum += y[idx] * x[idx];
    }

    __shared__ ValueType sdata[BLOCKSIZE];
    sdata[threadIdx.x] = sum;

    __syncthreads();

    if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
    if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
    if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
    if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
    if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
    if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();

    if(threadIdx.x == 0)
    {
        workspace[blockIdx.x] = sdata[0] + sdata[1];
    }
}

template <unsigned int BLOCKSIZE, typename ValueType>
__launch_bounds__(BLOCKSIZE)
__global__ void dot_kernel_part2(ValueType* workspace)
{
    __shared__ ValueType sdata[BLOCKSIZE];
    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();

    if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
    if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
    if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
    if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
    if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
    if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();

    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0] + sdata[1];
    }
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_DOT_IMPL_HPP