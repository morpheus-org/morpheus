/**
 * Morpheus_Spaces.hpp
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

#ifndef MORPHEUS_SPACES_HPP
#define MORPHEUS_SPACES_HPP

#include <Morpheus_CustomBackend.hpp>
#include <Morpheus_GenericBackend.hpp>

namespace Morpheus {
template <class ExecutionSpace, class MemorySpace, class BackendSpace>
struct Device {
  static_assert(Morpheus::has_execution_space<ExecutionSpace>::value,
                "ExecutionSpace does not have a valid execution_space trait");
  static_assert(Morpheus::has_memory_space<MemorySpace>::value,
                "MemorySpace does not have a valid memory_space trait");
  static_assert(Morpheus::is_custom_backend<BackendSpace>::value ||
                    Morpheus::is_generic_backend<BackendSpace>::value,
                "BackendSpace must be either a custom or generic backend.");
  using backend         = typename BackendSpace::backend;
  using execution_space = typename ExecutionSpace::execution_space;
  using memory_space    = typename MemorySpace::memory_space;
  using device_type     = Device<execution_space, memory_space, backend>;
};

template <typename S>
struct HostMirror {
 private:
  // If input execution space can access HostSpace then keep it.
  // Example: Kokkos::OpenMP can access, Kokkos::Cuda cannot
  enum {
    keep_exe = Kokkos::Impl::MemorySpaceAccess<
        typename S::execution_space::memory_space,
        Kokkos::HostSpace>::accessible
  };

  // If HostSpace can access memory space then keep it.
  // Example:  Cannot access Kokkos::CudaSpace, can access Kokkos::CudaUVMSpace
  enum {
    keep_mem =
        Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                        typename S::memory_space>::accessible
  };

  using wrapped_space =
      typename std::conditional<is_execution_space_v<S> || is_memory_space_v<S>,
                                Morpheus::GenericBackend<S>, S>::type;

 public:
  //  static_assert(check that S is not Kokkos::)
  using backend = typename std::conditional<
      keep_exe && keep_mem, wrapped_space,
      typename std::conditional<
          keep_mem,
          Morpheus::Device<Kokkos::HostSpace::execution_space,
                           typename wrapped_space::memory_space, wrapped_space>,
          Morpheus::HostSpace>::type>::type;
};

/**
 * @brief Checks if the given type \p T has a valid supported backend.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_backend {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              is_custom_backend_v<typename U::backend> ||
              is_generic_backend_v<typename U::backend>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_backend.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_backend_v = has_backend<T>::value;

}  // namespace Morpheus

#endif  // MORPHEUS_SPACES_HPP