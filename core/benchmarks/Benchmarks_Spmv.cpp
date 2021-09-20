/**
 * Benchmarks_Spmv.cpp
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
#include <Kokkos_Random.hpp>
#include <cstdlib>

#if defined(WITH_SERIAL)
using CustomSpace   = Kokkos::Serial;
using MorpheusSpace = Morpheus::Serial;
#elif defined(WITH_OPENMP)
using CustomSpace   = Kokkos::OpenMP;
using MorpheusSpace = Morpheus::OpenMP;
#elif defined(WITH_CUDA)
using CustomSpace   = Kokkos::Cuda;
using MorpheusSpace = Morpheus::Cuda;
#endif

using coo       = Morpheus::CooMatrix<double, int, CustomSpace>;
using csr       = Morpheus::CsrMatrix<double, int, CustomSpace>;
using dia       = Morpheus::DiaMatrix<double, int, CustomSpace>;
using dyn       = Morpheus::DynamicMatrix<double, int, CustomSpace>;
using vec       = Morpheus::vector<double, CustomSpace>;
using Generator = Kokkos::Random_XorShift64_Pool<CustomSpace>;

uint64_t reps       = 50;
uint64_t print_freq = 50;
uint64_t seed       = 0;

Morpheus::TimerPool timer;

template <typename Space, typename Matrix>
void spmv_bench(const Matrix& A, vec& x, vec& y,
                enum Morpheus::TimerPool::timer_id spmv, std::string fn_str) {
  uint64_t rep = 0;
  for (uint64_t s = seed; s < seed + reps; s++) {
    x.assign(Generator(s), 0, 100);
    y.assign(y.size(), 0);

    timer.start(spmv);
    Morpheus::multiply<Space>(A, x, y);
    Kokkos::fence();
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
    if (argc != 4) {
      std::cout << "Benchmarks_Spmv.cpp requires 3 input arguments to be given "
                   "at runtime (filename, seed, reps). Only received "
                << argc - 1 << "!" << std::endl;
      exit(-1);
    }

    std::string filename = argv[1];
    seed                 = atoi(argv[2]);
    reps                 = atoi(argv[3]);
    print_freq           = reps / 10;
    if (print_freq == 0) print_freq = 1;

    std::cout << "\nRunning Benchmarks_Spmv.cpp with:\n";
    std::cout << "\tFilename:\t" << filename << "\n";
    std::cout << "\tSeed:\t" << seed << "\n";
    std::cout << "\tReps:    \t" << reps << "\n\n";

    typename coo::HostMirror Aio;

    try {
      timer.start(timer.IO_READ);
      Morpheus::Io::read_matrix_market_file(Aio, filename);
      timer.stop(timer.IO_READ);
    } catch (Morpheus::NotImplementedException& e) {
      std::cerr << "Exception Raised:: " << e.what() << std::endl;
      exit(0);
    }

    vec x("x", Aio.ncols()), y("y", Aio.nrows());

    std::cout << "Starting experiment:" << std::endl;

    {
      coo A = Morpheus::create_mirror<CustomSpace>(Aio);
      Morpheus::copy(Aio, A);

      spmv_bench<CustomSpace>(A, x, y, timer.SPMV_COO_CUSTOM, "c_c_spmv<coo>");
      spmv_bench<MorpheusSpace>(A, x, y, timer.SPMV_COO_MORPHEUS,
                                "c_m_spmv<coo>");

      dyn Adyn;
      Adyn.activate(Morpheus::COO_FORMAT);
      Adyn = A;
      spmv_bench<CustomSpace>(Adyn, x, y, timer.SPMV_DYN_COO_CUSTOM,
                              "d_c_spmv<coo>");
      spmv_bench<MorpheusSpace>(Adyn, x, y, timer.SPMV_DYN_COO_MORPHEUS,
                                "d_m_spmv<coo>");
    }

    {
      typename csr::HostMirror Acsr_io;
      Morpheus::convert(Aio, Acsr_io);
      csr A = Morpheus::create_mirror<CustomSpace>(Acsr_io);
      Morpheus::copy(Acsr_io, A);

      spmv_bench<CustomSpace>(A, x, y, timer.SPMV_CSR_CUSTOM, "c_c_spmv<csr>");
      spmv_bench<MorpheusSpace>(A, x, y, timer.SPMV_CSR_MORPHEUS,
                                "c_m_spmv<csr>");

      dyn Adyn;
      Adyn.activate(Morpheus::CSR_FORMAT);
      Adyn = A;
      spmv_bench<CustomSpace>(Adyn, x, y, timer.SPMV_DYN_CSR_CUSTOM,
                              "d_c_spmv<csr>");
      spmv_bench<MorpheusSpace>(Adyn, x, y, timer.SPMV_DYN_CSR_MORPHEUS,
                                "d_m_spmv<csr>");
    }

    {
      typename dia::HostMirror Adia_io;
      Morpheus::convert(Aio, Adia_io);
      dia A = Morpheus::create_mirror<CustomSpace>(Adia_io);
      Morpheus::copy(Adia_io, A);

      spmv_bench<CustomSpace>(A, x, y, timer.SPMV_DIA_CUSTOM, "c_c_spmv<dia>");
      spmv_bench<MorpheusSpace>(A, x, y, timer.SPMV_DIA_MORPHEUS,
                                "c_m_spmv<dia>");

      dyn Adyn;
      Adyn.activate(Morpheus::DIA_FORMAT);
      Adyn = A;
      spmv_bench<CustomSpace>(Adyn, x, y, timer.SPMV_DYN_DIA_CUSTOM,
                              "d_c_spmv<dia>");
      spmv_bench<MorpheusSpace>(Adyn, x, y, timer.SPMV_DYN_DIA_MORPHEUS,
                                "d_m_spmv<dia>");
    }
  }

  timer.statistics();
  Morpheus::finalize();

  return 0;
}