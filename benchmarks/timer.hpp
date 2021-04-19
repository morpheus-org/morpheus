/**
 * timer.hpp
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

#ifndef MORPHEUS_BENCHMARK_TIMER_HPP
#define MORPHEUS_BENCHMARK_TIMER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <cfloat>
#include <iomanip>

#ifdef MORPHEUS_ENABLE_OPENMP
#include <omp.h>
using time_point = double;
#define omp_get_duration(t_elapse, t_start) \
  (t_elapse = omp_get_wtime() - t_start)
#else
#include <chrono>
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
#define omp_get_wtime() (std::chrono::high_resolution_clock::now())
#define omp_get_duration(t_elapse, t_start)                             \
  {                                                                     \
    std::chrono::duration<double> duration = omp_get_wtime() - t_start; \
    t_elapse                               = duration.count();          \
  }
#endif

namespace Morpheus {
namespace Impl {
class TimerInstance {
 private:
  time_point t_start;
  double t_sum;
  double t_max;
  double t_min;
  unsigned int active;
  unsigned int nsteps;

 public:
  TimerInstance()
      : t_sum(0.0), t_max(FLT_MIN), t_min(FLT_MAX), active(0), nsteps(0) {}

  void start() {
    t_start = omp_get_wtime();
    active  = 1;
    nsteps += 1;
  }

  void stop() {
    double t_elapse;

    if (active) {
      omp_get_duration(t_elapse, t_start);
      t_sum += t_elapse;
      t_max  = std::max(t_max, t_elapse);
      t_min  = std::min(t_min, t_elapse);
      active = 0;
    }
  }

  void statistics(const std::string &name) {
    if (nsteps != 0) {
      std::cout << std::setw(20) << name << "\t" << std::setw(10)
                << std::setprecision(7) << t_min << "\t" << std::setw(10)
                << std::setprecision(7) << t_max << "\t" << std::setw(10)
                << std::setprecision(7) << t_sum << "\t" << std::setw(20)
                << std::setprecision(10) << t_sum / static_cast<double>(nsteps)
                << "\t"
                << "(" << nsteps << "calls)" << std::endl;
    }
  }
};
}  // namespace Impl

class TimerPool {
 private:
  std::vector<Impl::TimerInstance> instances;
  std::vector<std::string> timer_name = {"Total",
                                         "I/O Read",
                                         "I/O Write",
                                         "Convert_COO_COO",
                                         "Convert_COO_CSR",
                                         "Convert_COO_DIA",
                                         "Convert_CSR_COO",
                                         "Convert_CSR_CSR",
                                         "Convert_CSR_DIA",
                                         "Convert_DIA_COO",
                                         "Convert_DIA_CSR",
                                         "Convert_DIA_DIA",
                                         "Convert_DYN_COO_COO",
                                         "Convert_DYN_COO_CSR",
                                         "Convert_DYN_COO_DIA",
                                         "Convert_DYN_CSR_COO",
                                         "Convert_DYN_CSR_CSR",
                                         "Convert_DYN_CSR_DIA",
                                         "Convert_DYN_DIA_COO",
                                         "Convert_DYN_DIA_CSR",
                                         "Convert_DYN_DIA_DIA",
                                         "Set_Vecs",
                                         "SpMv_COO",
                                         "SpMv_CSR",
                                         "SpMv_DIA",
                                         "SpMv_DYN_COO",
                                         "SpMv_DYN_CSR",
                                         "SpMv_DYN_DIA"};

 public:
  enum timer_id {
    TOTAL = 0,
    IO_READ,
    IO_WRITE,
    CONVERT_COO_COO,
    CONVERT_COO_CSR,
    CONVERT_COO_DIA,
    CONVERT_CSR_COO,
    CONVERT_CSR_CSR,
    CONVERT_CSR_DIA,
    CONVERT_DIA_COO,
    CONVERT_DIA_CSR,
    CONVERT_DIA_DIA,
    CONVERT_DYN_COO_COO,
    CONVERT_DYN_COO_CSR,
    CONVERT_DYN_COO_DIA,
    CONVERT_DYN_CSR_COO,
    CONVERT_DYN_CSR_CSR,
    CONVERT_DYN_CSR_DIA,
    CONVERT_DYN_DIA_COO,
    CONVERT_DYN_DIA_CSR,
    CONVERT_DYN_DIA_DIA,
    SET_VECS,
    SPMV_COO,
    SPMV_CSR,
    SPMV_DIA,
    SPMV_DYN_COO,
    SPMV_DYN_CSR,
    SPMV_DYN_DIA,
    NTIMERS /* This must be the last entry */
  };

  TimerPool() : instances(timer_id::NTIMERS){};

  void start(const int t_id) { instances[t_id].start(); }

  void stop(const int t_id) { instances[t_id].stop(); }

  void statistics() {
    std::cout << "\nTimer statistics:" << std::endl;
    std::cout << std::setw(20) << "Section"
              << "\t" << std::setw(10) << "tmin"
              << "\t" << std::setw(10) << "tmax"
              << "\t" << std::setw(10) << "total" << std::endl;

    for (int n = 0; n < timer_id::NTIMERS; n++) {
      instances[n].statistics(timer_name[n]);
    }
  }
};
}  // namespace Morpheus

#endif  // MORPHEUS_BENCHMARK_TIMER_HPP
