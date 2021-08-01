#include <impl/DenseVector/Kernels/Morpheus_Elementwise_Impl.hpp>

namespace Morpheus {
namespace Impl {
namespace Kernels {
// template <typename ValueType>
// __global__ void elementwise_kernel(const int n, const ValueType* x,
//                                    const ValueType* y, ValueType* xy) {
//   const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   xy[tid]       = x[tid] * y[tid];
// }

__global__ void elementwise_kernel(int n, const double* x, const double* y,
                                   double* xy) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  xy[tid]       = x[tid] * y[tid];
}
}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus
