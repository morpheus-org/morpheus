/**
 * Benchmarks_Copy.cpp
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

template <typename Fmt>
void deep(const Fmt& A, enum Morpheus::TimerPool::timer_id tidx,
          const std::string& fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    timer.start(tidx);
    auto B = Morpheus::create_mirror<ExecSpace>(A);
    Morpheus::copy(A, B);
    timer.stop(tidx);

    std::cout << "\tStep " << rep + 1 << "/" << reps << "\t" << fn_str
              << std::endl;
  }
}

template <typename Fmt>
void elem(const Fmt& A, enum Morpheus::TimerPool::timer_id tidx,
          const std::string& fn_str) {
  for (auto rep = 0; rep < reps; rep++) {
    timer.start(tidx);
    Fmt B;
    Morpheus::convert(A, B);
    timer.stop(tidx);

    std::cout << "\tStep " << rep + 1 << "/" << reps << "\t" << fn_str
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
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

        deep<coo>(A, timer.COPY_COO_DEEP, "deep<coo>");
        elem<coo>(A, timer.COPY_COO_ELEM, "elem<coo>");
      }

      {
        csr A;
        Morpheus::convert(Aio, A);

        deep<csr>(A, timer.COPY_CSR_DEEP, "deep<csr>");
        elem<csr>(A, timer.COPY_CSR_ELEM, "elem<csr>");
      }

      {
        try {
          dia A;
          Morpheus::convert(Aio, A);

          deep<dia>(A, timer.COPY_DIA_DEEP, "deep<dia>");
          elem<dia>(A, timer.COPY_DIA_ELEM, "elem<dia>");
        } catch (std::exception& e) {
          std::cerr << "Exception Raised:: " << e.what() << std::endl;
        }
      }
    }
  }

  timer.statistics();
  Morpheus::finalize();

  return 0;
}