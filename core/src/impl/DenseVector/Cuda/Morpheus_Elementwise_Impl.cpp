#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <impl/DenseVector/Kernels/Morpheus_Elementwise_Impl.hpp>
#include <impl/DenseVector/Cuda/Morpheus_Elementwise_Impl.hpp>

namespace Morpheus {
namespace Impl {

// template <typename ValueType, class... P>
void elementwise(const DenseVector<double, Kokkos::Cuda>& x,
                 const DenseVector<double, Kokkos::Cuda>& y,
                 DenseVector<double, Kokkos::Cuda>& xy) {
  const size_t BLOCK_SIZE = 256;
  const size_t NUM_BLOCKS = (x.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const double* x_ptr = x.data();
  const double* y_ptr = y.data();
  double* xy_ptr      = xy.data();

  //   Morpheus::Impl::Kernels::elementwise_kernel<ValueType>
  //       <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(x.size(), x_ptr, y_ptr, xy_ptr);
  Kernels::elementwise_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(x.size(), x_ptr,
                                                             y_ptr, xy_ptr);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA