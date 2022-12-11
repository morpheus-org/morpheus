/**
 * Morpheus_OpenMPUtils.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_OPENMP_UTILS_HPP
#define MORPHEUS_OPENMP_UTILS_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <omp.h>

namespace Morpheus {
namespace Impl {

template <typename T>
T _split_work(const T load, const T workers, const T worker_id) {
  const T unifload = load / workers;  // uniform distribution
  const T rem      = load - unifload * workers;
  T bound;

  //  round-robin assignment of the remaining work
  if (worker_id <= rem) {
    bound = (unifload + 1) * worker_id;
  } else {
    bound = (unifload + 1) * rem + unifload * (worker_id - rem);
  }

  return bound;
}

template <typename T>
void atomic_add(T* out, T val) {
#pragma omp atomic
  *out += val;
}

template <typename T = int>
T threads() {
  T t = 1;
#pragma omp parallel
  { t = omp_get_num_threads(); }

  return t;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_OPENMP_UTILS_HPP