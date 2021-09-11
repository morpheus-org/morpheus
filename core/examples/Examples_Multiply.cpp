/**
 * Examples_Multiply.cpp
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

using coo = Morpheus::CooMatrix<double, int, Kokkos::HostSpace>;
using dyn = Morpheus::DynamicMatrix<double, int, Kokkos::HostSpace>;
using vec = Morpheus::DenseVector<double, Kokkos::HostSpace>;

using serial = Kokkos::Serial;
#if defined(MORPHEUS_ENABLE_OPENMP)
using omp = Kokkos::OpenMP;
#endif

int main() {
  Morpheus::initialize();
  {
    coo A(4, 3, 6);
    vec x(3, 2), ya("ya", 4, 0);

    // initialize matrix entries
    A.row_indices[0]    = 0;
    A.column_indices[0] = 0;
    A.values[0]         = 10;
    A.row_indices[1]    = 0;
    A.column_indices[1] = 2;
    A.values[1]         = 20;
    A.row_indices[2]    = 2;
    A.column_indices[2] = 2;
    A.values[2]         = 30;
    A.row_indices[3]    = 3;
    A.column_indices[3] = 0;
    A.values[3]         = 40;
    A.row_indices[4]    = 3;
    A.column_indices[4] = 1;
    A.values[4]         = 50;
    A.row_indices[5]    = 3;
    A.column_indices[5] = 2;
    A.values[5]         = 60;

    Morpheus::print(A);
    Morpheus::print(x);

    Morpheus::multiply<serial>(A, x, ya);
    Morpheus::print(ya);

#if defined(MORPHEUS_ENABLE_OPENMP)
    dyn B(A);
    vec yb("yb", 4, 0);

    Morpheus::multiply<omp>(B, x, yb);
    Morpheus::print(yb);
#endif
  }
  Morpheus::finalize();
}