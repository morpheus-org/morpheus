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
#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/containers/vector.hpp>
#include <morpheus/algorithms/multiply.hpp>
#include <morpheus/algorithms/print.hpp>

using coo = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using csr = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using vec = Morpheus::DenseVector<double, Kokkos::Serial>;
int main() {
  Morpheus::initialize();
  {
    coo A(4, 3, 6);
    csr B(4, 3, 6);
    vec x(4, 2), ycoo("ycoo", 4, 0), ycsr("ycsr", 4, 0);

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

    B.row_offsets[0] = 0;  // first offset is always zero
    B.row_offsets[1] = 2;
    B.row_offsets[2] = 2;
    B.row_offsets[3] = 3;
    B.row_offsets[4] = 6;  // last offset is always num_entries

    B.column_indices[0] = 0;
    B.values[0]         = 10;
    B.column_indices[1] = 2;
    B.values[1]         = 20;
    B.column_indices[2] = 2;
    B.values[2]         = 30;
    B.column_indices[3] = 0;
    B.values[3]         = 40;
    B.column_indices[4] = 1;
    B.values[4]         = 50;
    B.column_indices[5] = 2;
    B.values[5]         = 60;
    Morpheus::print(A);
    Morpheus::print(B);

    Morpheus::multiply(A, x, ycoo);
    Morpheus::multiply(B, x, ycsr);

    Morpheus::print(ycoo);
    Morpheus::print(ycsr);
  }

  Morpheus::finalize();
}