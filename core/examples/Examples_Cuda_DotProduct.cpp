/**
 * Examples_Cuda_DotProduct.cpp
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

#include <Morpheus_Core.hpp>
#include <Morpheus_MirrorContainers.hpp>
#include <iostream>

#include <impl/DenseVector/Cuda/Morpheus_Elementwise_Impl.hpp>

using vec_serial = Morpheus::DenseVector<double, Kokkos::Serial>;
using vec_cuda   = Morpheus::DenseVector<double, Kokkos::Cuda>;

int main() {
  Morpheus::initialize();
  {
    try {
      vec_serial As(5, 2);
      vec_serial::HostMirror As_mirror = Morpheus::create_mirror(As);
      Morpheus::copy(As, As_mirror);
    } catch (std::runtime_error& e) {
      std::cerr << "vec_serial::HostMirror::Exception Raised:: " << e.what()
                << std::endl;
    }

    try {
      vec_cuda x(5, 2), y(5, 2), xy(5, 0);
      vec_cuda::HostMirror xy_mirror = Morpheus::create_mirror(xy);

      Morpheus::Impl::elementwise(x, y, xy);
      Morpheus::copy(xy, xy_mirror);
      Morpheus::print(xy_mirror);
    } catch (std::runtime_error& e) {
      std::cerr << "vec_cuda::HostMirror::Exception Raised:: " << e.what()
                << std::endl;
    }
  }
  Morpheus::finalize();
}