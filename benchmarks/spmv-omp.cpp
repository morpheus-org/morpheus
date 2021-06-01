/**
 * spmv-omp.cpp
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

using coo       = Morpheus::CooMatrix<double, int, Kokkos::OpenMP>;
using csr       = Morpheus::CsrMatrix<double, int, Kokkos::OpenMP>;
using dia       = Morpheus::DiaMatrix<double, int, Kokkos::OpenMP>;
using dyn       = Morpheus::DynamicMatrix<double, int, Kokkos::OpenMP>;
using vec       = Morpheus::vector<double, Kokkos::OpenMP>;
using exec      = typename Kokkos::OpenMP::execution_space;
using Generator = Kokkos::Random_XorShift64_Pool<Kokkos::OpenMP>;

Morpheus::TimerPool timer;

#define COO_FORMAT Morpheus::COO_FORMAT
#define CSR_FORMAT Morpheus::CSR_FORMAT
#define DIA_FORMAT Morpheus::DIA_FORMAT

template <typename Matrix>
void spmv_bench(const Matrix& A, enum Morpheus::TimerPool::timer_id spmv,
                uint64_t seed, int reps, int print_freq, std::string fn_str) {
  int rep = 0;
  for (uint64_t s = seed, cols = A.ncols(); s < seed + reps; s++) {
    timer.start(timer.SET_VECS);
    vec x("x", cols, Generator(s), 0, 100);
    vec y("y", x.size(), 0);
    timer.stop(timer.SET_VECS);

    timer.start(spmv);
    exec space;  // Run with OpenMP
    Morpheus::multiply(space, A, x, y);
    timer.stop(spmv);

    if (rep % print_freq == 0) {
      std::cout << "\t Step " << rep + 1 << "/" << reps << "\t" << fn_str
                << std::endl;
    }

    rep = rep + 1;
  }
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    std::string filename = argv[1];
    uint64_t seed        = atoi(argv[2]);
    uint64_t reps        = atoi(argv[3]);
    int print_freq       = reps / 10;
    if (print_freq == 0) print_freq = 1;

    std::cout << "\nRunning convert.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tSeed:\t" << seed << "\n";
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
      coo A(Aio);
      spmv_bench(A, timer.SPMV_COO, seed, reps, print_freq,
                 "concrete_spmv<coo>");
      dyn Adyn(A);
      spmv_bench(Adyn, timer.SPMV_DYN_COO, seed, reps, print_freq,
                 "dynamic_spmv<coo>");
    }

    {
      csr A(Aio);
      spmv_bench(A, timer.SPMV_CSR, seed, reps, print_freq,
                 "concrete_spmv<csr>");
      dyn Adyn(A);
      spmv_bench(Adyn, timer.SPMV_DYN_CSR, seed, reps, print_freq,
                 "dynamic_spmv<csr>");
    }

    {
      dia A(Aio);
      spmv_bench(A, timer.SPMV_DIA, seed, reps, print_freq,
                 "concrete_spmv<dia>");
      dyn Adyn(A);
      spmv_bench(Adyn, timer.SPMV_DYN_DIA, seed, reps, print_freq,
                 "dynamic_spmv<dia>");
    }
  }

  timer.statistics();

  Morpheus::finalize();
}