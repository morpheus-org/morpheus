/**
 * convert.cpp
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
#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/dia_matrix.hpp>

#include <morpheus/algorithms/print.hpp>
#include <morpheus/algorithms/convert.hpp>

using coo = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using csr = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using dia = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;

template <typename RefFmt, typename SrcFmt, typename DstFmt>
void run_ctors(RefFmt ref, std::string str) {
  std::cout << str + "::Begin" << std::endl;
  SrcFmt A(ref);
  DstFmt B = A;
  Morpheus::print(ref);
  Morpheus::print(B);
  std::cout << str + "::End\n" << std::endl;
}

template <typename RefFmt, typename SrcFmt, typename DstFmt>
void run(RefFmt ref, std::string str) {
  std::cout << str + "::Begin" << std::endl;
  SrcFmt A;
  DstFmt B;
  Morpheus::convert(ref, A);
  Morpheus::convert(A, B);
  Morpheus::print(ref);
  Morpheus::print(B);
  std::cout << str + "::End\n" << std::endl;
}

int main() {
  Morpheus::initialize();
  {
    coo Acoo(4, 3, 6);
    csr Acsr(4, 3, 6);
    dia Adia(4, 3, 6, 5);

    Acoo.row_indices[0]    = 0;
    Acoo.column_indices[0] = 0;
    Acoo.values[0]         = 10;
    Acoo.row_indices[1]    = 0;
    Acoo.column_indices[1] = 2;
    Acoo.values[1]         = 20;
    Acoo.row_indices[2]    = 2;
    Acoo.column_indices[2] = 2;
    Acoo.values[2]         = 30;
    Acoo.row_indices[3]    = 3;
    Acoo.column_indices[3] = 0;
    Acoo.values[3]         = 40;
    Acoo.row_indices[4]    = 3;
    Acoo.column_indices[4] = 1;
    Acoo.values[4]         = 50;
    Acoo.row_indices[5]    = 3;
    Acoo.column_indices[5] = 2;
    Acoo.values[5]         = 60;

    Acsr.row_offsets[0] = 0;  // first offset is always zero
    Acsr.row_offsets[1] = 2;
    Acsr.row_offsets[2] = 2;
    Acsr.row_offsets[3] = 3;
    Acsr.row_offsets[4] = 6;  // last offset is always num_entries

    Acsr.column_indices[0] = 0;
    Acsr.values[0]         = 10;
    Acsr.column_indices[1] = 2;
    Acsr.values[1]         = 20;
    Acsr.column_indices[2] = 2;
    Acsr.values[2]         = 30;
    Acsr.column_indices[3] = 0;
    Acsr.values[3]         = 40;
    Acsr.column_indices[4] = 1;
    Acsr.values[4]         = 50;
    Acsr.column_indices[5] = 2;
    Acsr.values[5]         = 60;

    // Diagonal offsets
    Adia.diagonal_offsets[0] = -3;
    Adia.diagonal_offsets[1] = -2;
    Adia.diagonal_offsets[2] = -1;
    Adia.diagonal_offsets[3] = 0;
    Adia.diagonal_offsets[4] = 2;

    // First Diagonal
    Adia.values(0, 0) = 40;
    Adia.values(0, 1) = -1;
    Adia.values(0, 2) = -1;
    // Second Diagonal
    Adia.values(1, 0) = 0;
    Adia.values(1, 1) = 50;
    Adia.values(1, 2) = -1;
    // Third Diagonal
    Adia.values(2, 0) = 0;
    Adia.values(2, 1) = 0;
    Adia.values(2, 2) = 60;
    // Main Diagonal
    Adia.values(3, 0) = 10;
    Adia.values(3, 1) = 0;
    Adia.values(3, 2) = 30;
    // Fifth Diagonal
    Adia.values(4, 0) = -1;
    Adia.values(4, 1) = -1;
    Adia.values(4, 2) = 20;

    {
      run<coo, coo, coo>(Acoo, "coo->coo->coo");
      run<coo, csr, coo>(Acoo, "coo->csr->coo");
      run<coo, dia, coo>(Acoo, "coo->dia->coo");
    }
    {
      run<csr, coo, csr>(Acoo, "csr->coo->csr");
      run<csr, csr, csr>(Acoo, "csr->csr->csr");
      run<csr, dia, csr>(Acoo, "csr->dia->csr");
    }
    {
      run<dia, coo, dia>(Acoo, "dia->coo->dia");
      run<dia, csr, dia>(Acoo, "dia->csr->dia");
      run<dia, dia, dia>(Acoo, "dia->dia->dia");
    }

    {
      run_ctors<coo, coo, coo>(Acoo, "coo->coo->coo::OO");
      run_ctors<coo, csr, coo>(Acoo, "coo->csr->coo::OO");
      run_ctors<coo, dia, coo>(Acoo, "coo->dia->coo::OO");
    }
    {
      run_ctors<csr, coo, csr>(Acoo, "csr->coo->csr::OO");
      run_ctors<csr, csr, csr>(Acoo, "csr->csr->csr::OO");
      run_ctors<csr, dia, csr>(Acoo, "csr->dia->csr::OO");
    }
    {
      run_ctors<dia, coo, dia>(Acoo, "dia->coo->dia::OO");
      run_ctors<dia, csr, dia>(Acoo, "dia->csr->dia::OO");
      run_ctors<dia, dia, dia>(Acoo, "dia->dia->dia::OO");
    }
  }
  Morpheus::finalize();
}