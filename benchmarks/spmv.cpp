/**
 * spmv.cpp
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
#include <morpheus/containers/vector.hpp>
#include <morpheus/algorithms/multiply.hpp>

#include "timer.hpp"
#include <Kokkos_Random.hpp>
#include <cstdlib>

using coo       = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using csr       = Morpheus::CsrMatrix<double, int, Kokkos::Serial>;
using dia       = Morpheus::DiaMatrix<double, int, Kokkos::Serial>;
using dyn       = Morpheus::DynamicMatrix<double, int, Kokkos::Serial>;
using vec       = Morpheus::vector<double, Kokkos::Serial>;
using exec      = typename Kokkos::Serial::execution_space;
using Generator = Kokkos::Random_XorShift64_Pool<Kokkos::Serial>;

Morpheus::TimerPool timer;

#define COO_FORMAT Morpheus::COO_FORMAT
#define CSR_FORMAT Morpheus::CSR_FORMAT
#define DIA_FORMAT Morpheus::DIA_FORMAT

template <typename Matrix>
void spmv_bench(const Matrix& A, const vec x, vec y,
                enum Morpheus::TimerPool::timer_id spmv) {
  timer.start(spmv);
  exec space;
  Morpheus::multiply(space, A, x, y);
  timer.stop(spmv);
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    std::string filename = argv[1];
    uint64_t seed        = atoi(argv[2]);
    uint64_t reps        = atoi(argv[3]);
    int print_freq       = reps / 10;

    std::cout << "\nRunning convert.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tSeed:\t" << seed << "\n";
    std::cout << "\tReps:    \t" << reps << "\n\n";

    coo Aio;

    timer.start(timer.IO_READ);
    Morpheus::Io::read_matrix_market_file(Aio, filename);
    timer.stop(timer.IO_READ);

    int rep = 0;
    std::cout << "Starting experiment:\n";
    for (uint64_t s = seed, cols = Aio.ncols(); s < seed + reps; s++) {
      timer.start(timer.SET_VECS);
      vec x("x", cols, Generator(s), 0, 100);
      vec y("y", x.size(), 0);
      timer.stop(timer.SET_VECS);

      {
        coo A(Aio);
        spmv_bench(A, x, y, timer.SPMV_COO);
      }

      {
        csr A(Aio);
        spmv_bench(A, x, y, timer.SPMV_CSR);
      }

      {
        dia A(Aio);
        spmv_bench(A, x, y, timer.SPMV_DIA);
      }

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tconcrete_spmv<coo,csr,dia>\n";
      }

      {
        dyn A(Aio);
        A.convert(COO_FORMAT);
        spmv_bench(A, x, y, timer.SPMV_DYN_COO);

        A.convert(CSR_FORMAT);
        spmv_bench(A, x, y, timer.SPMV_DYN_CSR);

        A.convert(DIA_FORMAT);
        spmv_bench(A, x, y, timer.SPMV_DYN_DIA);
      }

      if (rep % print_freq == 0) {
        std::cout << "\t Step " << rep << "/" << reps
                  << "\tdynamic_spmv<coo,csr,dia>\n";
      }
      rep = rep + 1;
    }
  }

  timer.statistics();

  Morpheus::finalize();
}