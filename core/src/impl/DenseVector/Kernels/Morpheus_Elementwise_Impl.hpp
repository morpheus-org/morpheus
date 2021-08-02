#ifndef MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP

#include <cuda.h>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template <typename ValueType>
__global__ void elementwise_kernel(int n, const ValueType* x,
                                   const ValueType* y, ValueType* xy) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  xy[tid]       = x[tid] * y[tid] + tid;
}

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP