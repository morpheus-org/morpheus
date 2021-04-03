/**
 * CooMatrix.cpp
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

#include <iostream>
#include <morpheus/core/core.hpp>
#include <morpheus/containers/coo_matrix.hpp>
#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/algorithms/print.hpp>

using coo_default = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using coo_omp     = Morpheus::CooMatrix<double, int, Kokkos::OpenMP>;

static_assert(
    std::is_same_v<coo_default::memory_space, Kokkos::Serial::memory_space>);
static_assert(std::is_same_v<coo_default::execution_space,
                             Kokkos::Serial::execution_space>);

static_assert(
    std::is_same_v<coo_omp::memory_space, Kokkos::OpenMP::memory_space>);
static_assert(
    std::is_same_v<coo_omp::execution_space, Kokkos::OpenMP::execution_space>);

static_assert(std::is_same_v<coo_default::memory_space, Kokkos::HostSpace>);
static_assert(std::is_same_v<coo_omp::memory_space, Kokkos::HostSpace>);
// Compiled with OpenMP, HostSpace Execution space same with is OpenMP's
// static_assert(std::is_same_v<coo_default::execution_space,
//                              Kokkos::HostSpace::execution_space>);
static_assert(std::is_same_v<coo_omp::execution_space,
                             Kokkos::HostSpace::execution_space>);

int main() {
  Morpheus::initialize();
  {
    coo_default A(4, 3, 6);
    coo_omp B(5, 5, 4);

    std::cout << std::is_same_v<coo_default::memory_space,
                                coo_omp::memory_space> << std::endl;
    std::cout << std::is_same_v<coo_default::execution_space,
                                coo_omp::execution_space> << std::endl;

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

    // A now represents the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]
    Morpheus::print(A);
  }

  {
    Morpheus::vector<double> x("DenseVector", 5);
    x[0] = 0.0;
    x[1] = 0.1;
    x[2] = 0.2;
    x[3] = 0.3;
    x[4] = 0.4;
    Morpheus::print(x);
  }
  {
    Morpheus::DynamicMatrix<double, int> B;

    Morpheus::print(B);
  }
  {
    Morpheus::CooMatrix<double, int> A(4, 3, 6);
    Morpheus::DynamicMatrix<double, int> B(A);  // shallow copy
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

    Morpheus::print(B);
  }

  {
    Morpheus::CooMatrix<double, int> A("A", 4, 3, 6);
    Morpheus::CooMatrix<double, int> B(A);   // Copy constructor (shallow)
    Morpheus::CooMatrix<double, int> C = B;  // Copy assignment (shallow)

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
    Morpheus::print(B);
    Morpheus::print(C);

    std::cout << A.name() << "\n" << B.name() << "\n" << C.name() << std::endl;
  }
  Morpheus::finalize();
}