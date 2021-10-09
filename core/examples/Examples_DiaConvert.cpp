/**
 * Examples_DiaConvert.cpp
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
#include <algorithm>

using coo    = Morpheus::CooMatrix<double, int, Kokkos::HostSpace>;
using dia    = Morpheus::DiaMatrix<double, int, Kokkos::HostSpace>;
using vec    = Morpheus::DenseVector<double, Kokkos::HostSpace>;
using serial = Kokkos::Serial;

// Medium matrix
// [ 1, -1, -3,  0,  0]
// [-2,  5,  0,  0,  0]
// [ 0,  0,  4,  6,  4]
// [-4,  0,  2,  7,  0]
// [ 0,  8,  0,  0, -5]
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

int main() {
  Morpheus::initialize();
  {
    coo A = build_medium_coo(), from_dia;
    dia B = build_medium_dia(), from_coo;

    // Morpheus::convert(B, from_dia);
    Morpheus::convert(A, from_coo);

    Morpheus::print(B);
    Morpheus::print(from_coo);
  }

  Morpheus::finalize();
}
