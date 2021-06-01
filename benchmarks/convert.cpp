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
#include <morpheus/algorithms/copy.hpp>

#include <morpheus/algorithms/print.hpp>

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
void concrete_convert(const SrcFmt& A, enum Morpheus::TimerPool::timer_id tidx,
                      int reps, int print_freq, std::string fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    DstFmt B;

    timer.start(tidx);
    Morpheus::copy(A, B);
    timer.stop(tidx);

    if (rep % print_freq == 0) {
      std::cout << "\t Step " << rep + 1 << "/" << reps << "\t" << fn_str
                << std::endl;
    }
  }
}

template <typename SrcFmt>
void to_dynamic_convert(const SrcFmt& A, enum Morpheus::formats_e dstfmt,
                        enum Morpheus::TimerPool::timer_id tidx, int reps,
                        int print_freq, std::string fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    dyn B;
    timer.start(tidx);
    B.activate(dstfmt);
    Morpheus::copy(A, B);
    timer.stop(tidx);

    if (rep % print_freq == 0) {
      std::cout << "\t Step " << rep + 1 << "/" << reps << "\t" << fn_str
                << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    std::string filename = argv[1];
    int reps             = atoi(argv[2]);
    int print_freq       = reps / 10;
    if (print_freq == 0) print_freq = 1;

    std::cout << "\nRunning convert.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tReps:    \t" << reps << "\n" << std::endl;

    coo Aio;

    try {
      timer.start(timer.IO_READ);
      Morpheus::Io::read_matrix_market_file(Aio, filename);
      timer.stop(timer.IO_READ);
    } catch (Morpheus::NotImplementedException& e) {
      std::cerr << "Exception Raised:: " << e.what() << std::endl;
      exit(0);
    }

    std::cout << "Starting experiment:" << std::endl;

    {
      coo A(Aio);
      concrete_convert<coo, coo>(A, timer.CONVERT_COO_COO, reps, print_freq,
                                 "concrete_convert<coo, coo>");
      concrete_convert<coo, csr>(A, timer.CONVERT_COO_CSR, reps, print_freq,
                                 "concrete_convert<coo, csr>");
      concrete_convert<coo, dia>(A, timer.CONVERT_COO_DIA, reps, print_freq,
                                 "concrete_convert<coo, dia>");

      to_dynamic_convert<coo>(A, COO_FORMAT, timer.CONVERT_DYN_COO_COO, reps,
                              print_freq, "to_dynamic_convert<coo>[COO->COO]");
      to_dynamic_convert<coo>(A, CSR_FORMAT, timer.CONVERT_DYN_COO_CSR, reps,
                              print_freq, "to_dynamic_convert<coo>[COO->CSR]");
      to_dynamic_convert<coo>(A, DIA_FORMAT, timer.CONVERT_DYN_COO_DIA, reps,
                              print_freq, "to_dynamic_convert<coo>[COO->DIA]");
    }

    {
      csr A(Aio);
      concrete_convert<csr, coo>(A, timer.CONVERT_CSR_COO, reps, print_freq,
                                 "concrete_convert<csr, coo>");
      concrete_convert<csr, csr>(A, timer.CONVERT_CSR_CSR, reps, print_freq,
                                 "concrete_convert<csr, csr>");
      concrete_convert<csr, dia>(A, timer.CONVERT_CSR_DIA, reps, print_freq,
                                 "concrete_convert<csr, dia>");

      to_dynamic_convert<csr>(A, COO_FORMAT, timer.CONVERT_DYN_CSR_COO, reps,
                              print_freq, "to_dynamic_convert<csr>[CSR->COO]");
      to_dynamic_convert<csr>(A, CSR_FORMAT, timer.CONVERT_DYN_CSR_CSR, reps,
                              print_freq, "to_dynamic_convert<csr>[CSR->CSR]");
      to_dynamic_convert<csr>(A, DIA_FORMAT, timer.CONVERT_DYN_CSR_DIA, reps,
                              print_freq, "to_dynamic_convert<csr>[CSR->DIA]");
    }

    {
      dia A(Aio);
      concrete_convert<dia, coo>(A, timer.CONVERT_DIA_COO, reps, print_freq,
                                 "concrete_convert<dia, coo>");
      concrete_convert<dia, csr>(A, timer.CONVERT_DIA_CSR, reps, print_freq,
                                 "concrete_convert<dia, csr>");
      concrete_convert<dia, dia>(A, timer.CONVERT_DIA_DIA, reps, print_freq,
                                 "concrete_convert<dia, dia>");

      to_dynamic_convert<dia>(A, COO_FORMAT, timer.CONVERT_DYN_DIA_COO, reps,
                              print_freq, "to_dynamic_convert<dia>[DIA->COO]");
      to_dynamic_convert<dia>(A, CSR_FORMAT, timer.CONVERT_DYN_DIA_CSR, reps,
                              print_freq, "to_dynamic_convert<dia>[DIA->CSR]");
      to_dynamic_convert<dia>(A, DIA_FORMAT, timer.CONVERT_DYN_DIA_DIA, reps,
                              print_freq, "to_dynamic_convert<dia>[DIA->DIA]");
    }

    timer.statistics();
  }
  Morpheus::finalize();
}
