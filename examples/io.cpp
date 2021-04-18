/**
 * io.cpp
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

#include <morpheus/algorithms/print.hpp>
#include <morpheus/algorithms/convert.hpp>

#include <morpheus/containers/coo_matrix.hpp>
#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/dia_matrix.hpp>
#include <morpheus/containers/vector.hpp>

#include <morpheus/io/matrix_market.hpp>

using coo = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using csr = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using dia = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;
using vec = Morpheus::DenseVector<double, Kokkos::Serial>;

int main() {
  Morpheus::initialize();
  {
    vec x;
    std::string ifilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2_c.mtx";
    std::string ofilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2_c_out.mtx";

    Morpheus::Io::read_matrix_market_file(x, ifilename);
    Morpheus::Io::write_matrix_market_file(x, ofilename);
    Morpheus::print(x);
  }
  {
    coo A;
    std::string ifilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2.mtx";
    std::string ofilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2_out_coo.mtx";

    Morpheus::Io::read_matrix_market_file(A, ifilename);
    Morpheus::Io::write_matrix_market_file(A, ofilename);
    Morpheus::print(A);
  }
  {
    csr A;
    coo B;
    std::string ifilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2.mtx";
    std::string ofilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2_out_csr.mtx";

    Morpheus::Io::read_matrix_market_file(A, ifilename);
    Morpheus::Io::write_matrix_market_file(A, ofilename);
    Morpheus::convert(A, B);
    Morpheus::print(B);
  }
  {
    dia A;
    coo B;
    std::string ifilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2.mtx";
    std::string ofilename =
        "/Volumes/PhD/Code/Projects/morpheus/data/lp_kb2/lp_kb2_out_dia.mtx";

    Morpheus::Io::read_matrix_market_file(A, ifilename);
    Morpheus::Io::write_matrix_market_file(A, ofilename);
    Morpheus::convert(A, B);
    Morpheus::print(B);
  }
  Morpheus::finalize();
  return 0;
}