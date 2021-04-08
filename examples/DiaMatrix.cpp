/**
 * DiaMatrix.cpp
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
using dia = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;
using vec = Morpheus::DenseVector<double, Kokkos::Serial>;
using ser = typename Kokkos::Serial::execution_space;

int main() {
  Morpheus::initialize();
  {
    dia B(7, 5, 12, 3);
    // [ 1,  0, 13,  0,  0],
    // [ 0,  2,  0, 14,  0],
    // [ 0,  0,  3,  0, 15],
    // [ 6,  0,  0,  4,  0],
    // [ 0,  7,  0,  0,  5],
    // [ 0,  0,  8,  0,  0],
    // [ 0,  0,  0,  9,  0]

    // Diagonal offsets
    B.diagonal_offsets[0] = 0;
    B.diagonal_offsets[1] = -3;
    B.diagonal_offsets[2] = 2;
    // First Diagonal
    B.values(0, 0) = 1;
    B.values(0, 1) = 2;
    B.values(0, 2) = 3;
    B.values(0, 3) = 4;
    B.values(0, 4) = 5;
    // Second Diagonal
    B.values(1, 0) = 6;
    B.values(1, 1) = 7;
    B.values(1, 2) = 8;
    B.values(1, 3) = 9;
    B.values(1, 4) = 0;
    // Third Diagonal
    B.values(2, 0) = 0;
    B.values(2, 1) = 0;
    B.values(2, 2) = 13;
    B.values(2, 3) = 14;
    B.values(2, 4) = 15;

    Morpheus::print(B);
  }

  {
    dia B(4, 3, 6, 5);
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // Diagonal offsets
    B.diagonal_offsets[0] = 0;
    B.diagonal_offsets[1] = -3;
    B.diagonal_offsets[2] = -2;
    B.diagonal_offsets[3] = -1;
    B.diagonal_offsets[4] = 2;
    // First Diagonal
    B.values(0, 0) = 10;
    B.values(0, 1) = 0;
    B.values(0, 2) = 30;
    // Second Diagonal
    B.values(1, 0) = 40;
    B.values(1, 1) = -1;
    B.values(1, 2) = -1;
    // Third Diagonal
    B.values(2, 0) = 0;
    B.values(2, 1) = 50;
    B.values(2, 2) = -1;
    // Fourth Diagonal
    B.values(3, 0) = 0;
    B.values(3, 1) = 0;
    B.values(3, 2) = 60;
    // Fifth Diagonal
    B.values(4, 0) = -1;
    B.values(4, 1) = -1;
    B.values(4, 2) = 20;

    std::cout << "Stats:: B(" << B.nrows() << "," << B.ncols() << ")"
              << "\tB.values(" << B.values.nrows() << "," << B.values.ncols()
              << ")" << std::endl;
    for (int i = 0; i < B.values.nrows(); i++) {
      for (int j = 0; j < B.values.ncols(); j++) {
        std::cout << B.values(i, j) << "\t";
      }
      std::cout << std::endl;
    }

    Morpheus::print(B);
  }

  {
    //    [10 20 00]
    //    [00 00 30]
    //    [40 00 50]
    //    [00 60 00]
    coo A(4, 3, 6);
    dia B(4, 3, 6, 3);
    // initialize matrix entries
    A.row_indices[0]    = 0;
    A.column_indices[0] = 0;
    A.values[0]         = 10;
    A.row_indices[1]    = 0;
    A.column_indices[1] = 1;
    A.values[1]         = 20;
    A.row_indices[2]    = 1;
    A.column_indices[2] = 2;
    A.values[2]         = 30;
    A.row_indices[3]    = 2;
    A.column_indices[3] = 0;
    A.values[3]         = 40;
    A.row_indices[4]    = 2;
    A.column_indices[4] = 2;
    A.values[4]         = 50;
    A.row_indices[5]    = 3;
    A.column_indices[5] = 1;
    A.values[5]         = 60;
    // Diagonal offsets
    B.diagonal_offsets[0] = 0;
    B.diagonal_offsets[1] = -2;
    B.diagonal_offsets[2] = 1;
    // First Diagonal
    B.values(0, 0) = 10;
    B.values(0, 1) = 0;
    B.values(0, 2) = 50;
    // Second Diagonal
    B.values(1, 0) = 40;
    B.values(1, 1) = 60;
    B.values(1, 2) = -1;
    // Third Diagonal
    B.values(2, 0) = -1;
    B.values(2, 1) = 20;
    B.values(2, 2) = 30;

    vec x(3, 0), ya("ya", 4, 0), yb("yb", 4, 0);
    ser space;
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    Morpheus::multiply(space, A, x, ya);
    Morpheus::multiply(space, B, x, yb);

    Morpheus::print(ya);
    Morpheus::print(yb);
  }

  {
    coo A(4, 3, 6);
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

    dia B(4, 3, 6, 5);
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // Diagonal offsets
    B.diagonal_offsets[0] = 0;
    B.diagonal_offsets[1] = -3;
    B.diagonal_offsets[2] = -2;
    B.diagonal_offsets[3] = -1;
    B.diagonal_offsets[4] = 2;
    // First Diagonal
    B.values(0, 0) = 10;
    B.values(0, 1) = 0;
    B.values(0, 2) = 30;
    // Second Diagonal
    B.values(1, 0) = 40;
    B.values(1, 1) = -1;
    B.values(1, 2) = -1;
    // Third Diagonal
    B.values(2, 0) = 0;
    B.values(2, 1) = 50;
    B.values(2, 2) = -1;
    // Fourth Diagonal
    B.values(3, 0) = 0;
    B.values(3, 1) = 0;
    B.values(3, 2) = 60;
    // Fifth Diagonal
    B.values(4, 0) = -1;
    B.values(4, 1) = -1;
    B.values(4, 2) = 20;

    vec x(3, 0), ya("ya", 4, 0), yb("yb", 4, 0);
    ser space;
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    Morpheus::multiply(space, A, x, ya);
    Morpheus::multiply(space, B, x, yb);

    Morpheus::print(ya);
    Morpheus::print(yb);
  }

  Morpheus::finalize();
}