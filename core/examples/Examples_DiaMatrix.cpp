/**
 * Examples_DiaMatrix.cpp
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
#include <iostream>

using coo    = Morpheus::CooMatrix<double, int, Kokkos::HostSpace>;
using dia    = Morpheus::DiaMatrix<double, int, Kokkos::HostSpace>;
using vec    = Morpheus::DenseVector<double, Kokkos::HostSpace>;
using serial = typename Kokkos::Serial;

// Large matrix
// [ 1,  0, 13,  0,  0],
// [ 0,  2,  0, 14,  0],
// [ 0,  0,  3,  0, 15],
// [ 6,  0,  0,  4,  0],
// [ 0,  7,  0,  0,  5],
// [ 0,  0,  8,  0,  0],
// [ 0,  0,  0,  9,  0]

// Medium matrix
// [ 1, -1, -3,  0,  0]
// [-2,  5,  0,  0,  0]
// [ 0,  0,  4,  6,  4]
// [-4,  0,  2,  7,  0]
// [ 0,  8,  0,  0, -5]

// Many Diagonals Matrix
//    [10  0 20]
//    [ 0  0  0]
//    [ 0  0 30]
//    [40 50 60]

// Simple Matrix
//    [10 20 00]
//    [00 00 30]
//    [40 00 50]
//    [00 60 00]

coo build_large_coo();
dia build_large_dia();
coo build_medium_coo();
dia build_medium_dia();
coo build_many_diag_coo();
dia build_many_diag_dia();
coo build_simple_coo();
dia build_simple_dia();

int main() {
  Morpheus::initialize();
  {
    coo A = build_large_coo(), A_from_dia;
    dia B = build_large_dia(), B_from_coo;

    vec x(5, 0), ya("ycoo", 7, 0), yb("ydia", 7, 0);

    x(0) = 1;
    x(1) = 2;
    x(2) = 3;
    x(3) = 4;
    x(4) = 5;

    Morpheus::multiply<serial>(A, x, ya);
    Morpheus::multiply<serial>(B, x, yb);

    Morpheus::convert(A, B_from_coo);  // should be same with B
    Morpheus::convert(B, A_from_dia);  // should be same with A

    std::cout << "============ Checking build_large ============" << std::endl;
    Morpheus::print(ya);
    Morpheus::print(yb);

    Morpheus::print(A);
    Morpheus::print(A_from_dia);
    Morpheus::print(B);
    Morpheus::print(B_from_coo);

    std::cout << "==============================================" << std::endl;
  }

  {
    coo A = build_medium_coo(), A_from_dia;
    dia B = build_medium_dia(), B_from_coo;
    vec x(5, 0), ya("ycoo", 5, 0), yb("ydia", 5, 0);

    x(0) = 1;
    x(1) = 2;
    x(2) = 3;
    x(3) = 4;
    x(4) = 5;

    Morpheus::multiply<serial>(A, x, ya);
    Morpheus::multiply<serial>(B, x, yb);

    Morpheus::convert(A, B_from_coo);  // should be same with B
    Morpheus::convert(B, A_from_dia);  // should be same with A

    std::cout << "============ Checking build_medium ============\n";
    Morpheus::print(ya);
    Morpheus::print(yb);

    Morpheus::print(A);
    Morpheus::print(A_from_dia);
    Morpheus::print(B);
    Morpheus::print(B_from_coo);

    std::cout << "==============================================\n";
  }

  {
    coo A = build_many_diag_coo(), A_from_dia;
    dia B = build_many_diag_dia(), B_from_coo;

    vec x(3, 0), ya("ycoo", 4, 0), yb("ydia", 4, 0);

    x(0) = 1;
    x(1) = 2;
    x(2) = 3;

    Morpheus::multiply<serial>(A, x, ya);
    Morpheus::multiply<serial>(B, x, yb);

    Morpheus::convert(A, B_from_coo);  // should be same with B
    Morpheus::convert(B, A_from_dia);  // should be same with A

    std::cout << "============ Checking build_many_diag ============\n";
    Morpheus::print(ya);
    Morpheus::print(yb);

    Morpheus::print(A);
    Morpheus::print(A_from_dia);
    Morpheus::print(B);
    Morpheus::print(B_from_coo);

    std::cout << "==============================================\n";
  }

  {
    coo A = build_simple_coo(), A_from_dia;
    dia B = build_simple_dia(), B_from_coo;

    vec x(3, 0), ya("ycoo", 4, 0), yb("ydia", 4, 0);

    x(0) = 1;
    x(1) = 2;
    x(2) = 3;

    Morpheus::multiply<serial>(A, x, ya);
    Morpheus::multiply<serial>(B, x, yb);

    Morpheus::convert(A, B_from_coo);  // should be same with B
    Morpheus::convert(B, A_from_dia);  // should be same with A

    std::cout << "============ Checking build_simple ============\n";
    Morpheus::print(ya);
    Morpheus::print(yb);

    Morpheus::print(A);
    Morpheus::print(A_from_dia);
    Morpheus::print(B);
    Morpheus::print(B_from_coo);

    std::cout << "==============================================\n";
  }

  Morpheus::finalize();
}

dia build_large_dia() {
  dia mat(7, 5, 12, 3);
  // Diagonal offsets
  mat.diagonal_offsets(0) = -3;
  mat.diagonal_offsets(1) = 0;
  mat.diagonal_offsets(2) = 2;
  // First Diagonal
  mat.values(0, 0) = -1;
  mat.values(1, 0) = -1;
  mat.values(2, 0) = -1;
  mat.values(3, 0) = 6;
  mat.values(4, 0) = 7;
  mat.values(5, 0) = 8;
  mat.values(6, 0) = 9;
  // Second Diagonal
  mat.values(0, 1) = 1;
  mat.values(1, 1) = 2;
  mat.values(2, 1) = 3;
  mat.values(3, 1) = 4;
  mat.values(4, 1) = 5;
  mat.values(5, 1) = -2;
  mat.values(6, 1) = -2;
  // Third Diagonal
  mat.values(0, 2) = 13;
  mat.values(1, 2) = 14;
  mat.values(2, 2) = 15;
  mat.values(3, 2) = -33;
  mat.values(4, 2) = -33;
  mat.values(5, 2) = -33;
  mat.values(6, 2) = -33;

  return mat;
}

coo build_large_coo() {
  coo mat(7, 5, 12);
  // initialize matrix entries
  mat.row_indices(0)    = 0;
  mat.column_indices(0) = 0;
  mat.values(0)         = 1;

  mat.row_indices(1)    = 0;
  mat.column_indices(1) = 2;
  mat.values(1)         = 13;

  mat.row_indices(2)    = 1;
  mat.column_indices(2) = 1;
  mat.values(2)         = 2;

  mat.row_indices(3)    = 1;
  mat.column_indices(3) = 3;
  mat.values(3)         = 14;

  mat.row_indices(4)    = 2;
  mat.column_indices(4) = 2;
  mat.values(4)         = 3;

  mat.row_indices(5)    = 2;
  mat.column_indices(5) = 4;
  mat.values(5)         = 15;

  mat.row_indices(6)    = 3;
  mat.column_indices(6) = 0;
  mat.values(6)         = 6;

  mat.row_indices(7)    = 3;
  mat.column_indices(7) = 3;
  mat.values(7)         = 4;

  mat.row_indices(8)    = 4;
  mat.column_indices(8) = 1;
  mat.values(8)         = 7;

  mat.row_indices(9)    = 4;
  mat.column_indices(9) = 4;
  mat.values(9)         = 5;

  mat.row_indices(10)    = 5;
  mat.column_indices(10) = 2;
  mat.values(10)         = 8;

  mat.row_indices(11)    = 6;
  mat.column_indices(11) = 3;
  mat.values(11)         = 9;

  return mat;
}

dia build_medium_dia() {
  dia mat(5, 5, 13, 5);
  // Diagonal offsets
  mat.diagonal_offsets(0) = -3;
  mat.diagonal_offsets(1) = -1;
  mat.diagonal_offsets(2) = 0;
  mat.diagonal_offsets(3) = 1;
  mat.diagonal_offsets(4) = 2;
  // First Diagonal
  mat.values(0, 0) = -33;
  mat.values(1, 0) = -33;
  mat.values(2, 0) = -33;
  mat.values(3, 0) = -4;
  mat.values(4, 0) = 8;
  // Second Diagonal
  mat.values(0, 1) = -33;
  mat.values(1, 1) = -2;
  mat.values(2, 1) = 0;
  mat.values(3, 1) = 2;
  mat.values(4, 1) = 0;
  // Third Diagonal
  mat.values(0, 2) = 1;
  mat.values(1, 2) = 5;
  mat.values(2, 2) = 4;
  mat.values(3, 2) = 7;
  mat.values(4, 2) = -5;
  // Fourth Diagonal
  mat.values(0, 3) = -1;
  mat.values(1, 3) = 0;
  mat.values(2, 3) = 6;
  mat.values(3, 3) = 0;
  mat.values(4, 3) = -33;
  // Fifth Diagonal
  mat.values(0, 4) = -3;
  mat.values(1, 4) = 0;
  mat.values(2, 4) = 4;
  mat.values(3, 4) = -33;
  mat.values(4, 4) = -33;

  return mat;
}

coo build_medium_coo() {
  coo mat(5, 5, 13);
  // initialize matrix entries
  mat.row_indices(0)    = 0;
  mat.column_indices(0) = 0;
  mat.values(0)         = 1;

  mat.row_indices(1)    = 0;
  mat.column_indices(1) = 1;
  mat.values(1)         = -1;

  mat.row_indices(2)    = 0;
  mat.column_indices(2) = 2;
  mat.values(2)         = -3;

  mat.row_indices(3)    = 1;
  mat.column_indices(3) = 0;
  mat.values(3)         = -2;

  mat.row_indices(4)    = 1;
  mat.column_indices(4) = 1;
  mat.values(4)         = 5;

  mat.row_indices(5)    = 2;
  mat.column_indices(5) = 2;
  mat.values(5)         = 4;

  mat.row_indices(6)    = 2;
  mat.column_indices(6) = 3;
  mat.values(6)         = 6;

  mat.row_indices(7)    = 2;
  mat.column_indices(7) = 4;
  mat.values(7)         = 4;

  mat.row_indices(8)    = 3;
  mat.column_indices(8) = 0;
  mat.values(8)         = -4;

  mat.row_indices(9)    = 3;
  mat.column_indices(9) = 2;
  mat.values(9)         = 2;

  mat.row_indices(10)    = 3;
  mat.column_indices(10) = 3;
  mat.values(10)         = 7;

  mat.row_indices(11)    = 4;
  mat.column_indices(11) = 1;
  mat.values(11)         = 8;

  mat.row_indices(12)    = 4;
  mat.column_indices(12) = 4;
  mat.values(12)         = -5;

  return mat;
}

dia build_many_diag_dia() {
  dia mat(4, 3, 6, 5);
  // Diagonal offsets
  mat.diagonal_offsets(0) = -3;
  mat.diagonal_offsets(1) = -2;
  mat.diagonal_offsets(2) = -1;
  mat.diagonal_offsets(3) = 0;
  mat.diagonal_offsets(4) = 2;
  // First Diagonal
  mat.values(0, 0) = -1;
  mat.values(1, 0) = -1;
  mat.values(2, 0) = -1;
  mat.values(3, 0) = 40;
  // Second Diagonal
  mat.values(0, 1) = -2;
  mat.values(1, 1) = -2;
  mat.values(2, 1) = 0;
  mat.values(3, 1) = 50;
  // Third Diagonal
  mat.values(0, 2) = -3;
  mat.values(1, 2) = 0;
  mat.values(2, 2) = 0;
  mat.values(3, 2) = 60;
  // Fourth Diagonal
  mat.values(0, 3) = 10;
  mat.values(1, 3) = 0;
  mat.values(2, 3) = 30;
  mat.values(3, 3) = -4;
  // Fifth Diagonal
  mat.values(0, 4) = 20;
  mat.values(1, 4) = -5;
  mat.values(2, 4) = -5;
  mat.values(3, 4) = -5;

  return mat;
}

coo build_many_diag_coo() {
  coo mat(4, 3, 6);
  // initialize matrix entries
  mat.row_indices(0)    = 0;
  mat.column_indices(0) = 0;
  mat.values(0)         = 10;
  mat.row_indices(1)    = 0;
  mat.column_indices(1) = 2;
  mat.values(1)         = 20;
  mat.row_indices(2)    = 2;
  mat.column_indices(2) = 2;
  mat.values(2)         = 30;
  mat.row_indices(3)    = 3;
  mat.column_indices(3) = 0;
  mat.values(3)         = 40;
  mat.row_indices(4)    = 3;
  mat.column_indices(4) = 1;
  mat.values(4)         = 50;
  mat.row_indices(5)    = 3;
  mat.column_indices(5) = 2;
  mat.values(5)         = 60;

  return mat;
}

dia build_simple_dia() {
  dia mat(4, 3, 6, 3);

  // Diagonal offsets
  mat.diagonal_offsets(0) = -2;
  mat.diagonal_offsets(1) = 0;
  mat.diagonal_offsets(2) = 1;
  // First Diagonal
  mat.values(0, 0) = -1;
  mat.values(1, 0) = -1;
  mat.values(2, 0) = 40;
  mat.values(3, 0) = 60;
  // Second Diagonal
  mat.values(0, 1) = 10;
  mat.values(1, 1) = 0;
  mat.values(2, 1) = 50;
  mat.values(3, 1) = -2;
  // Third Diagonal
  mat.values(0, 2) = 20;
  mat.values(1, 2) = 30;
  mat.values(2, 2) = -3;
  mat.values(3, 2) = -3;

  return mat;
}

coo build_simple_coo() {
  coo mat(4, 3, 6);
  // initialize matrix entries
  mat.row_indices(0)    = 0;
  mat.column_indices(0) = 0;
  mat.values(0)         = 10;
  mat.row_indices(1)    = 0;
  mat.column_indices(1) = 1;
  mat.values(1)         = 20;
  mat.row_indices(2)    = 1;
  mat.column_indices(2) = 2;
  mat.values(2)         = 30;
  mat.row_indices(3)    = 2;
  mat.column_indices(3) = 0;
  mat.values(3)         = 40;
  mat.row_indices(4)    = 2;
  mat.column_indices(4) = 2;
  mat.values(4)         = 50;
  mat.row_indices(5)    = 3;
  mat.column_indices(5) = 1;
  mat.values(5)         = 60;

  return mat;
}