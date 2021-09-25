/**
 * Benchmarks_Convert.cpp
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

#include "timer.hpp"
#include <cstdlib>

using ExecSpace = Kokkos::Serial;

using coo = Morpheus::CooMatrix<double, int, ExecSpace>;
using csr = Morpheus::CsrMatrix<double, int, ExecSpace>;
using dia = Morpheus::DiaMatrix<double, int, ExecSpace>;
using dyn = Morpheus::DynamicMatrix<double, int, ExecSpace>;

auto reps = 50;
Morpheus::TimerPool timer;

template <typename SrcFmt, typename DstFmt>
void convert_c(const SrcFmt& A, enum Morpheus::TimerPool::timer_id tidx,
               const std::string& fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    DstFmt B;

    timer.start(tidx);
    Morpheus::convert(A, B);
    timer.stop(tidx);

    std::cout << "\tStep " << rep + 1 << "/" << reps << "\t" << fn_str
              << std::endl;
  }
}

template <typename SrcFmt>
void convert_d(const SrcFmt& A, enum Morpheus::formats_e dstfmt,
               enum Morpheus::TimerPool::timer_id tidx,
               const std::string& fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    dyn B;
    B.activate(dstfmt);

    timer.start(tidx);
    Morpheus::convert(A, B);
    timer.stop(tidx);

    std::cout << "\tStep " << rep + 1 << "/" << reps << "\t" << fn_str
              << std::endl;
  }
}

template <typename SrcFmt>
void convert_in_place(const SrcFmt& A, enum Morpheus::formats_e dstfmt,
                      enum Morpheus::TimerPool::timer_id tidx,
                      const std::string& fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    dyn B;
    B.activate(A.format_enum());
    B = A;

    timer.start(tidx);
    Morpheus::convert(B, dstfmt);
    timer.stop(tidx);

    std::cout << "\tStep " << rep + 1 << "/" << reps << "\t" << fn_str
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    if (argc != 3) {
      std::cout
          << "Benchmarks_Convert.cpp requires 2 input arguments to be given "
             "at runtime (filename, reps). Only received "
          << argc - 1 << "!" << std::endl;
      exit(-1);
    }

    std::string filename = argv[1];
    reps                 = atoi(argv[2]);

    std::cout << "\nRunning Benchmarks_Convert.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tReps:    \t" << reps << "\n\n";

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
      coo A;
      Morpheus::convert(Aio, A);

      convert_c<coo, coo>(A, timer.CONVERT_COO_COO, "convert_c<coo, coo>");
      convert_c<coo, csr>(A, timer.CONVERT_COO_CSR, "convert_c<coo, csr>");
      try {
        convert_c<coo, dia>(A, timer.CONVERT_COO_DIA, "convert_c<coo, dia>");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }

      convert_d<coo>(A, Morpheus::COO_FORMAT, timer.CONVERT_DYN_COO_COO,
                     "convert_d<coo>[COO->COO]");
      convert_d<coo>(A, Morpheus::CSR_FORMAT, timer.CONVERT_DYN_COO_CSR,
                     "convert_d<coo>[COO->CSR]");
      try {
        convert_d<coo>(A, Morpheus::DIA_FORMAT, timer.CONVERT_DYN_COO_DIA,
                       "convert_d<coo>[COO->DIA]");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }

      convert_in_place<coo>(A, Morpheus::COO_FORMAT, timer.CONVERT_IN_COO_COO,
                            "convert_in_place<coo>[COO->COO]");
      convert_in_place<coo>(A, Morpheus::CSR_FORMAT, timer.CONVERT_IN_COO_CSR,
                            "convert_in_place<coo>[COO->CSR]");
      try {
        convert_in_place<coo>(A, Morpheus::DIA_FORMAT, timer.CONVERT_IN_COO_DIA,
                              "convert_in_place<coo>[COO->DIA]");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }
    }

    {
      csr A;
      Morpheus::convert(Aio, A);

      convert_c<csr, coo>(A, timer.CONVERT_CSR_COO, "convert_c<csr, coo>");
      convert_c<csr, csr>(A, timer.CONVERT_CSR_CSR, "convert_c<csr, csr>");
      try {
        convert_c<csr, dia>(A, timer.CONVERT_CSR_DIA, "convert_c<csr, dia>");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }

      convert_d<csr>(A, Morpheus::COO_FORMAT, timer.CONVERT_DYN_CSR_COO,
                     "convert_d<csr>[CSR->COO]");
      convert_d<csr>(A, Morpheus::CSR_FORMAT, timer.CONVERT_DYN_CSR_CSR,
                     "convert_d<csr>[CSR->CSR]");
      try {
        convert_d<csr>(A, Morpheus::DIA_FORMAT, timer.CONVERT_DYN_CSR_DIA,
                       "convert_d<csr>[CSR->DIA]");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }

      convert_in_place<csr>(A, Morpheus::COO_FORMAT, timer.CONVERT_IN_CSR_COO,
                            "convert_in_place<csr>[CSR->COO]");
      convert_in_place<csr>(A, Morpheus::CSR_FORMAT, timer.CONVERT_IN_CSR_CSR,
                            "convert_in_place<csr>[CSR->CSR]");
      try {
        convert_in_place<csr>(A, Morpheus::DIA_FORMAT, timer.CONVERT_IN_CSR_DIA,
                              "convert_in_place<csr>[CSR->DIA]");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }
    }

    {
      try {
        dia A;
        Morpheus::convert(Aio, A);

        convert_c<dia, coo>(A, timer.CONVERT_DIA_COO, "convert_c<dia, coo>");
        convert_c<dia, csr>(A, timer.CONVERT_DIA_CSR, "convert_c<dia, csr>");
        convert_c<dia, dia>(A, timer.CONVERT_DIA_DIA, "convert_c<dia, dia>");

        convert_d<dia>(A, Morpheus::COO_FORMAT, timer.CONVERT_DYN_DIA_COO,
                       "convert_d<dia>[DIA->COO]");
        convert_d<dia>(A, Morpheus::CSR_FORMAT, timer.CONVERT_DYN_DIA_CSR,
                       "convert_d<dia>[DIA->CSR]");
        convert_d<dia>(A, Morpheus::DIA_FORMAT, timer.CONVERT_DYN_DIA_DIA,
                       "convert_d<dia>[DIA->DIA]");

        convert_in_place<dia>(A, Morpheus::COO_FORMAT, timer.CONVERT_IN_DIA_COO,
                              "convert_in_place<dia>[DIA->COO]");
        convert_in_place<dia>(A, Morpheus::CSR_FORMAT, timer.CONVERT_IN_DIA_CSR,
                              "convert_in_place<dia>[DIA->CSR]");
        convert_in_place<dia>(A, Morpheus::DIA_FORMAT, timer.CONVERT_IN_DIA_DIA,
                              "convert_in_place<dia>[DIA->DIA]");
      } catch (std::exception& e) {
        std::cerr << "Exception Raised:: " << e.what() << std::endl;
      }
    }
  }

  timer.statistics();
  Morpheus::finalize();

  return 0;
}