/**
 * Examples_Cuda.cpp
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
#include <impl/DenseVector/Cuda/Morpheus_Elementwise_Impl.hpp>

#include <iostream>

using vec_serial = Morpheus::DenseVector<double, Kokkos::Serial>;
using vec_cuda   = Morpheus::DenseVector<double, Kokkos::Cuda>;
using coo_serial = Morpheus::CooMatrix<double, Kokkos::Serial>;
using coo_cuda   = Morpheus::CooMatrix<double, Kokkos::Cuda>;

int main() {
  Morpheus::initialize();
  {  // Elementwise vec-vec on cuda
    {
      vec_cuda x(5, 2), y(5, 4), xy(5);
      Morpheus::Impl::elementwise(x, y, xy);
      auto xy_h = Morpheus::create_mirror_container(xy);
      Morpheus::copy(xy, xy_h);
      Morpheus::print(xy_h);
    }
    // Elementwise vec-vec from host to cuda
    {
      vec_serial x(5, 2), y(5, 4), xy(5);

      auto x_d  = Morpheus::create_mirror_container<Kokkos::Cuda>(x);
      auto y_d  = Morpheus::create_mirror_container<Kokkos::Cuda>(y);
      auto xy_d = Morpheus::create_mirror_container<Kokkos::Cuda>(x);

      Morpheus::copy(x, x_d);
      Morpheus::copy(y, y_d);

      Morpheus::Impl::elementwise(x_d, y_d, xy_d);
      Morpheus::copy(xy_d, xy);
      Morpheus::print(xy);
    }
    // SpMV COO with CUDA
    {
      coo_serial A(2, 3, 4);
      vec_serial x(3, 2), y(2);

      A.row_indices[0]    = 0;
      A.column_indices[0] = 0;
      A.values[0]         = 4;
      A.row_indices[1]    = 0;
      A.column_indices[1] = 1;
      A.values[1]         = 3;
      A.row_indices[2]    = 1;
      A.column_indices[2] = 1;
      A.values[2]         = 5;
      A.row_indices[3]    = 1;
      A.column_indices[3] = 2;
      A.values[3]         = 1;

      Morpheus::multiply<Kokkos::Serial>(A, x, y);
      // !TODO: create_mirror_container_copy that creates the mirror and
      // also performs a deep copy
      auto A_d = Morpheus::create_mirror_container<Kokkos::Cuda>(A);
      auto x_d = Morpheus::create_mirror_container<Kokkos::Cuda>(x);
      auto y_d = Morpheus::create_mirror_container<Kokkos::Cuda>(y);
      Morpheus::copy(A, A_d);
      Morpheus::copy(x, x_d);

      Morpheus::multiply<Kokkos::Cuda>(A_d, x_d, y_d);
      auto y_res = Morpheus::create_mirror_container(y_d);
      Morpheus::copy(y_d, y_res);

      Morpheus::print(y_res);
      Morpheus::print(y);
    }
  }
  Morpheus::finalize();

  return 0;
}