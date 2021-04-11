/**
 * DynamicCopy.cpp
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

#include <morpheus/algorithms/print.hpp>
#include <morpheus/algorithms/copy.hpp>
#include <morpheus/core/exceptions.hpp>

using coo_ser = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using dyn_ser = Morpheus::DynamicMatrix<double, int, Kokkos::Serial>;
using dyn_omp = Morpheus::DynamicMatrix<double, int, Kokkos::OpenMP>;
using coo_omp = Morpheus::CooMatrix<double, int, Kokkos::OpenMP>;

int main() {
  Morpheus::initialize();
  {
    coo_ser A(3, 2, 4), B(2, 1, 2);
    dyn_ser C(A);
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

    B.row_indices[0]    = 0;
    B.column_indices[0] = 0;
    B.values[0]         = 10;
    B.row_indices[1]    = 0;
    B.column_indices[1] = 2;
    B.values[1]         = 20;

    Morpheus::copy(B, C);
    A.row_indices[0]    = 1;
    A.column_indices[0] = 1;
    A.values[0]         = -5;
    Morpheus::print(A);
    Morpheus::print(B);
    Morpheus::print(C);
  }
  Morpheus::finalize();
}