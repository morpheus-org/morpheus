#ifndef MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP

#include <Kokkos_Core.hpp>

namespace Morpheus {
namespace Impl {
namespace Kernels {
// template <typename ValueType>
// __global__ void elementwise_kernel(const int n, const ValueType* x,
//                                    const ValueType* y, ValueType* xy);
__global__ void elementwise_kernel(int n, const double* x, const double* y,
                                   double* xy);
}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_KERNELS_ELEMENTWISE_IMPL_HPP