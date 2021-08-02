#include <impl/DenseVector/Kernels/Morpheus_Elementwise_Impl.hpp>

namespace Morpheus {
namespace Impl {
namespace Kernels {

template __global__ void elementwise_kernel(int n, const double* x,
                                            const double* y, double* xy);

template __global__ void elementwise_kernel(int n, const float* x,
                                            const float* y, float* xy);

}  // namespace Kernels
}  // namespace Impl
}  // namespace Morpheus
