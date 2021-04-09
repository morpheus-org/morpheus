/**
 * Copy.cpp
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
using coo_omp = Morpheus::CooMatrix<double, int, Kokkos::OpenMP>;
using csr     = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using dia     = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;
using mat     = Morpheus::DenseMatrix<double, Kokkos::Serial>;
using vec     = Morpheus::DenseVector<double, Kokkos::Serial>;

int main() {
  Morpheus::initialize();
  {
    //   Same size
    vec src("src", 3), dst("dst", 3);
    src(0) = 1.2;
    src(1) = 1.3;
    src(2) = 1.4;
    Morpheus::print(src);
    dst(0) = 2.2;
    dst(1) = 2.3;
    dst(2) = 2.4;
    Morpheus::print(dst);

    Morpheus::copy(src, dst);
    src(0) = -1.2;
    dst(0) = 2.2;
    Morpheus::print(src);
    Morpheus::print(dst);
  }

  {
    //   Different size: src < dst
    vec src("src", 2), dst("dst", 4);
    src(0) = 1.2;
    src(1) = 1.3;
    Morpheus::print(src);
    dst(0) = 2.2;
    dst(1) = 2.3;
    dst(2) = 2.4;
    dst(3) = 2.5;
    Morpheus::print(dst);

    try {
      Morpheus::copy(src, dst);
    } catch (Morpheus::RuntimeException& e) {
      std::cerr << "Exception Raised:: " << e.what() << std::endl;
    }

    Morpheus::print(src);
    Morpheus::print(dst);
  }

  {
    //   Different size: src > dst
    vec src("src", 4), dst("dst", 2);
    src(0) = 1.2;
    src(1) = 1.3;
    src(2) = 1.4;
    src(3) = 1.5;
    Morpheus::print(src);
    dst(0) = 2.2;
    dst(1) = 2.3;
    Morpheus::print(dst);

    try {
      Morpheus::copy(src, dst);
    } catch (Morpheus::RuntimeException& e) {
      std::cerr << "Exception Raised:: " << e.what() << std::endl;
    }

    Morpheus::print(src);
    Morpheus::print(dst);
  }
  {
    // Coo serial to Coo serial
    coo_ser A(4, 3, 3);
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
    Morpheus::print(A);

    coo_ser B(4, 3, 3);
    B.row_indices[0]    = 1;
    B.column_indices[0] = 2;
    B.values[0]         = 12;
    B.row_indices[1]    = 2;
    B.column_indices[1] = 3;
    B.values[1]         = 23;
    B.row_indices[2]    = 3;
    B.column_indices[2] = 2;
    B.values[2]         = 32;
    Morpheus::print(B);

    Morpheus::copy(A, B);
    A.row_indices[0]    = 0;
    A.column_indices[0] = 1;
    A.values[0]         = 01;

    B.row_indices[0]    = 1;
    B.column_indices[0] = 2;
    B.values[0]         = 12;
    Morpheus::print(A);
    Morpheus::print(B);
  }
  {
    // Coo serial to Coo omp
    coo_ser A(4, 3, 3);
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
    Morpheus::print(A);

    coo_omp B(4, 3, 3);
    B.row_indices[0]    = 1;
    B.column_indices[0] = 2;
    B.values[0]         = 12;
    B.row_indices[1]    = 2;
    B.column_indices[1] = 3;
    B.values[1]         = 23;
    B.row_indices[2]    = 3;
    B.column_indices[2] = 2;
    B.values[2]         = 32;
    Morpheus::print(B);

    Morpheus::copy(A, B);
    A.row_indices[0]    = 0;
    A.column_indices[0] = 1;
    A.values[0]         = 01;

    B.row_indices[0]    = 1;
    B.column_indices[0] = 2;
    B.values[0]         = 12;
    Morpheus::print(A);
    Morpheus::print(B);
  }
  {
    mat A(3, 2, 1.0), B(3, 2, 2.0);

    Morpheus::copy(A, B);
    A(0, 0) = -1.0;
    B(0, 0) = -2.0;

    Morpheus::print(A);
    Morpheus::print(B);
  }
  {
    csr A(3, 2, 4), B(3, 2, 4);
    A.row_offsets[0] = 0;  // first offset is always zero
    A.row_offsets[1] = 1;
    A.row_offsets[2] = 3;
    A.row_offsets[4] = 4;  // last offset is always num_entries

    A.column_indices[0] = 0;
    A.values[0]         = 10;
    A.column_indices[1] = 0;
    A.values[1]         = 20;
    A.column_indices[2] = 2;
    A.values[2]         = 30;
    A.column_indices[3] = 1;
    A.values[3]         = 40;

    B.row_offsets[0] = 0;  // first offset is always zero
    B.row_offsets[1] = 1;
    B.row_offsets[2] = 3;
    B.row_offsets[3] = 4;  // last offset is always num_entries

    B.column_indices[0] = 1;
    B.values[0]         = -10;
    B.column_indices[1] = 1;
    B.values[1]         = -20;
    B.column_indices[2] = 2;
    B.values[2]         = -30;
    B.column_indices[3] = 0;
    B.values[3]         = 40;

    Morpheus::copy(A, B);
    A.column_indices[0] = 2;
    A.values[0]         = 5;
    B.column_indices[0] = 0;
    B.values[0]         = -5;

    Morpheus::print(A);
    Morpheus::print(B);
  }
  {
    csr A(4, 3, 6), B(3, 2, 4);
    A.row_offsets[0] = 0;  // first offset is always zero
    A.row_offsets[1] = 2;
    A.row_offsets[2] = 2;
    A.row_offsets[3] = 3;
    A.row_offsets[4] = 6;  // last offset is always num_entries

    A.column_indices[0] = 0;
    A.values[0]         = 10;
    A.column_indices[1] = 2;
    A.values[1]         = 20;
    A.column_indices[2] = 2;
    A.values[2]         = 30;
    A.column_indices[3] = 0;
    A.values[3]         = 40;
    A.column_indices[4] = 1;
    A.values[4]         = 50;
    A.column_indices[5] = 2;
    A.values[5]         = 60;

    B.row_offsets[0] = 0;  // first offset is always zero
    B.row_offsets[1] = 1;
    B.row_offsets[2] = 3;
    B.row_offsets[3] = 4;  // last offset is always num_entries

    B.column_indices[0] = 1;
    B.values[0]         = -10;
    B.column_indices[1] = 1;
    B.values[1]         = -20;
    B.column_indices[2] = 2;
    B.values[2]         = -30;
    B.column_indices[3] = 0;
    B.values[3]         = 40;

    Morpheus::copy(A, B);
    A.column_indices[0] = 2;
    A.values[0]         = 5;
    B.column_indices[0] = 0;
    B.values[0]         = -5;

    Morpheus::print(A);
    Morpheus::print(B);
  }
  {
    dia A(3, 4, 5, 2);
    // [ 1,  0, 13,  0],
    // [ 0,  2,  0, 14],
    // [ 0,  0,  3,  0]
    // Diagonal offsets
    A.diagonal_offsets[0] = 0;
    A.diagonal_offsets[1] = 2;
    // First Diagonal
    A.values(0, 0) = 1;
    A.values(0, 1) = 2;
    A.values(0, 2) = 3;
    // Second Diagonal
    A.values(1, 0) = 0;
    A.values(1, 1) = 13;
    A.values(1, 2) = 14;

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

    Morpheus::copy(A, B);
    Morpheus::print(A);
    Morpheus::print(B);
  }
  Morpheus::finalize();
}