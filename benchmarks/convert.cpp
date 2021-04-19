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

#include <morpheus/io/matrix_market.hpp>
#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/algorithms/convert.hpp>

#include "timer.hpp"
#include <cstdlib>

using coo = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using csr = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using dia = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;
using dyn = Morpheus::DynamicMatrix<double, int, Kokkos::Serial>;

#define COO_FORMAT Morpheus::COO_FORMAT
#define CSR_FORMAT Morpheus::CSR_FORMAT
#define DIA_FORMAT Morpheus::DIA_FORMAT

Morpheus::TimerPool timer;

template <typename SrcFmt, typename DstFmt>
void concrete_convert(coo ref, enum Morpheus::TimerPool::timer_id tidx) {
  SrcFmt A(ref);
  timer.start(tidx);
  DstFmt B = A;
  timer.stop(tidx);
}

void dynamic_convert(coo ref, enum Morpheus::formats_e srcfmt,
                     enum Morpheus::formats_e dstfmt,
                     enum Morpheus::TimerPool::timer_id tidx) {
  dyn A(ref);
  A.convert(srcfmt);

  timer.start(tidx);
  A.convert(dstfmt);
  timer.stop(tidx);
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    std::string filename = argv[1];
    uint64_t reps        = atoi(argv[2]);
    int print_freq       = reps / 10;

    std::cout << "\nRunning convert.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tReps:    \t" << reps << "\n\n";

    coo Aio;

    timer.start(timer.IO_READ);
    Morpheus::Io::read_matrix_market_file(Aio, filename);
    timer.stop(timer.IO_READ);

    std::cout << "Starting experiment:\n";
    for (uint64_t rep = 0; rep < reps; rep++) {
      concrete_convert<coo, coo>(Aio, timer.CONVERT_COO_COO);
      concrete_convert<coo, csr>(Aio, timer.CONVERT_COO_CSR);
      concrete_convert<coo, dia>(Aio, timer.CONVERT_COO_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tconcrete_convert<coo>\n";
      }

      concrete_convert<csr, coo>(Aio, timer.CONVERT_CSR_COO);
      concrete_convert<csr, csr>(Aio, timer.CONVERT_CSR_CSR);
      concrete_convert<csr, dia>(Aio, timer.CONVERT_CSR_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tconcrete_convert<csr>\n";
      }

      concrete_convert<dia, coo>(Aio, timer.CONVERT_DIA_COO);
      concrete_convert<dia, csr>(Aio, timer.CONVERT_DIA_CSR);
      concrete_convert<dia, dia>(Aio, timer.CONVERT_DIA_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tconcrete_convert<dia>\n";
      }

      dynamic_convert(Aio, COO_FORMAT, COO_FORMAT, timer.CONVERT_DYN_COO_COO);
      dynamic_convert(Aio, COO_FORMAT, CSR_FORMAT, timer.CONVERT_DYN_COO_CSR);
      dynamic_convert(Aio, COO_FORMAT, DIA_FORMAT, timer.CONVERT_DYN_COO_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tdynamic_convert<coo>\n";
      }

      dynamic_convert(Aio, CSR_FORMAT, COO_FORMAT, timer.CONVERT_DYN_CSR_COO);
      dynamic_convert(Aio, CSR_FORMAT, CSR_FORMAT, timer.CONVERT_DYN_CSR_CSR);
      dynamic_convert(Aio, CSR_FORMAT, DIA_FORMAT, timer.CONVERT_DYN_CSR_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tdynamic_convert<csr>\n";
      }

      dynamic_convert(Aio, DIA_FORMAT, COO_FORMAT, timer.CONVERT_DYN_DIA_COO);
      dynamic_convert(Aio, DIA_FORMAT, CSR_FORMAT, timer.CONVERT_DYN_DIA_CSR);
      dynamic_convert(Aio, DIA_FORMAT, DIA_FORMAT, timer.CONVERT_DYN_DIA_DIA);

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tdynamic_convert<dia>\n";
      }
    }

    timer.statistics();
  }
  Morpheus::finalize();
}