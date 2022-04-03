/**
 * Morpheus_KokkosSpace.hpp
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

#ifndef MORPHEUS_KOKKOSSPACE_HPP
#define MORPHEUS_KOKKOSSPACE_HPP

#include <Morpheus_TypeTraits.hpp>

namespace Morpheus {

// Wrapper of different Kokkos execution spaces to pass around and distinguish
// between custom and Kokkos Kernels
template <typename Space>
struct KokkosSpace {
  static_assert(is_execution_space_v<Space>,
                "Space needs to be a valid Kokkos ExecutionSpace!");
  using kokkos_space = KokkosSpace;
  using type         = Space;

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using device_type     = typename Space::device_type;
};

#if defined(MORPHEUS_ENABLE_SERIAL)
using Serial = Morpheus::KokkosSpace<Kokkos::Serial>;
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
using OpenMP = Morpheus::KokkosSpace<Kokkos::OpenMP>;
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
using Cuda = Morpheus::KokkosSpace<Kokkos::Cuda>;
#endif

}  // namespace Morpheus

#endif  // MORPHEUS_KOKKOSSPACE_HPP