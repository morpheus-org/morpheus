/**
 * Morpheus_SpaceTraits.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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
#ifndef MORPHEUS_SPACETRAITS_HPP
#define MORPHEUS_SPACETRAITS_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Morpheus {

/**
 * \defgroup space_traits Space Traits
 * \brief Various tools for examining the different types of spaces available
 * and relationships between them during compile-time.
 * \ingroup type_traits
 *
 */

/**
 * @brief Checks if the given type \p T is a valid supported memory space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_same<U, Kokkos::HostSpace>::value ||
#if defined(MORPHEUS_ENABLE_CUDA)
                                  std::is_same<U, Kokkos::CudaSpace>::value ||
#elif defined(MORPHEUS_ENABLE_HIP)
                                  std::is_same<U, Kokkos::HIPSpace>::value ||
#endif
                                  false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_memory_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_memory_space_v = is_memory_space<T>::value;

/**
 * @brief Checks if the given type \p T has a valid supported memory space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              is_memory_space_v<typename U::memory_space>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_memory_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_memory_space_v = has_memory_space<T>::value;

/**
 * @brief Checks if the given type \p T is a valid supported execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_same<U, Kokkos::DefaultHostExecutionSpace>::value ||
              std::is_same<U, Kokkos::DefaultExecutionSpace>::value ||
#if defined(MORPHEUS_ENABLE_SERIAL)
              std::is_same<U, Kokkos::Serial>::value ||
#endif
#if defined(MORPHEUS_ENABLE_OPENMP)
              std::is_same<U, Kokkos::OpenMP>::value ||
#endif
#if defined(MORPHEUS_ENABLE_CUDA)
              std::is_same<U, Kokkos::Cuda>::value ||
#elif defined(MORPHEUS_ENABLE_HIP)
              std::is_same<U, Kokkos::Experimental::HIP>::value ||
#endif
              false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_execution_space_v = is_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has a valid supported execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_execution_space_v<typename U::execution_space>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_execution_space_v = has_execution_space<T>::value;

/**
 * @brief Checks if the two types are in the same valid supported memory space
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_same_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<is_memory_space<U1>::value &&
                              is_memory_space<U2>::value &&
                              std::is_same<U1, U2>::value>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_same_memory_space.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_memory_space_v =
    is_same_memory_space<T1, T2>::value;

/**
 * @brief Checks if the two types have the same valid supported memory space
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class has_same_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<is_same_memory_space<
          typename U1::memory_space, typename U2::memory_space>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_same_memory_space.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool has_same_memory_space_v =
    has_same_memory_space<T1, T2>::value;

/**
 * @brief Checks if the given type \p T is a valid supported Host memory space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_host_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
                           std::is_same<U, Kokkos::HostSpace>::value ||
#endif
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_host_memory_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_host_memory_space_v = is_host_memory_space<T>::value;

/**
 * @brief Checks if the given type \p T has a valid supported Host memory space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_host_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*,
                   typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
                       is_host_memory_space<typename U::memory_space>::value ||
#endif
                       false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_host_memory_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_host_memory_space_v = has_host_memory_space<T>::value;

/**
 * @brief Checks if the given type \p T is a supported Host execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_host_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_same<U, Kokkos::DefaultHostExecutionSpace>::value ||
#if defined(MORPHEUS_ENABLE_SERIAL)
              std::is_same<U, Kokkos::Serial>::value ||
#endif
#if defined(MORPHEUS_ENABLE_OPENMP)
              std::is_same<U, Kokkos::OpenMP>::value ||
#endif
              false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_host_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_host_execution_space_v =
    is_host_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has a supported Host execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_host_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<is_host_execution_space_v<
                           typename U::execution_space>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_host_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_host_execution_space_v =
    has_host_execution_space<T>::value;

#if defined(MORPHEUS_ENABLE_SERIAL)
/**
 * @brief Checks if the given type \p T is a Serial execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_serial_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL)
                           std::is_same<U, Kokkos::Serial>::value ||
#endif  // MORPHEUS_ENABLE_SERIAL
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_serial_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_serial_execution_space_v =
    is_serial_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has a Serial execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_serial_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL)
              is_serial_execution_space<typename U::execution_space>::value ||
#endif  // MORPHEUS_ENABLE_SERIAL
              false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_serial_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_serial_execution_space_v =
    has_serial_execution_space<T>::value;
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
/**
 * @brief Checks if the given type \p T is an OpenMP execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_openmp_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_OPENMP)
                           std::is_same<U, Kokkos::OpenMP>::value ||
#endif  // MORPHEUS_ENABLE_OPENMP
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_openmp_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_openmp_execution_space_v =
    is_openmp_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has an OpenMP execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_openmp_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_OPENMP)
              is_openmp_execution_space<typename U::execution_space>::value ||
#endif  // MORPHEUS_ENABLE_OPENMP
              false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_openmp_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_openmp_execution_space_v =
    has_openmp_execution_space<T>::value;
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
/**
 * @brief Checks if the given type \p T is a Cuda execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_cuda_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_CUDA)
                           std::is_same<U, Kokkos::Cuda>::value ||
#endif  // MORPHEUS_ENABLE_CUDA
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_cuda_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_cuda_execution_space_v =
    is_cuda_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has a Cuda execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_cuda_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_CUDA)
                           Morpheus::is_cuda_execution_space<
                               typename U::execution_space>::value ||
#endif  // MORPHEUS_ENABLE_CUDA
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_cuda_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_cuda_execution_space_v =
    has_cuda_execution_space<T>::value;
#endif  // MORPHEUS_ENABLE_CUDA

#if defined(MORPHEUS_ENABLE_HIP)
/**
 * @brief Checks if the given type \p T is a HIP execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_hip_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*, typename std::enable_if<
#if defined(MORPHEUS_ENABLE_HIP)
                           std::is_same<U, Kokkos::HIP>::value ||
#endif  // MORPHEUS_ENABLE_HIP
                           false>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_hip_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_hip_execution_space_v =
    is_hip_execution_space<T>::value;

/**
 * @brief Checks if the given type \p T has a HIP execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_hip_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*,
                   typename std::enable_if<is_hip_execution_space<
                       typename U::execution_space>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_hip_execution_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_hip_execution_space_v =
    has_hip_execution_space<T>::value;
#endif  // MORPHEUS_ENABLE_HIP

/*! \cond */
namespace Impl {
template <typename ExecSpace, typename... Ts>
struct has_access;

template <class T1, class T2>
struct has_access<T1, T2> {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          has_execution_space_v<U1> && has_memory_space_v<U2> &&
          Kokkos::SpaceAccessibility<typename U1::execution_space,
                                     typename U2::memory_space>::accessible>::
          type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

template <typename ExecSpace, typename T, typename... Ts>
struct has_access<ExecSpace, T, Ts...> {
  static const bool value =
      has_access<ExecSpace, T>::value && has_access<ExecSpace, Ts...>::value;
};

}  // namespace Impl
/*! \endcond */

/**
 * @brief Takes arbitrary number of containers and checks if \p ExecSpace has
 * access to the memory space of all of them. Note that each container must have
 * a valid \p memory_space trait.
 *
 * @tparam ExecSpace Execution Space
 * @tparam Ts Aribtrary number of containers
 */
template <typename ExecSpace, typename... Ts>
struct has_access {
  static const bool value = Impl::has_access<ExecSpace, Ts...>::value;
};

/**
 * @brief Short-hand to \p has_access.
 *
 * @tparam ExecSpace Execution Space
 * @tparam Ts Aribtrary number of containers
 */
template <typename ExecSpace, typename... Ts>
inline constexpr bool has_access_v = has_access<ExecSpace, Ts...>::value;

/**
 * @brief Checks if the given type \p T is a valid supported space.
 *
 * @tparam T Type passed for check.
 */

template <typename T>
struct is_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(U*,
                   typename std::enable_if<
                       Kokkos::is_space<typename U::execution_space>::value ||
                       Kokkos::is_space<typename U::memory_space>::value ||
                       Kokkos::is_space<typename U::device_type>::value ||
                       Kokkos::is_space<U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_space_v = is_space<T>::value;

/*! \} end of space_traits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_SPACETRAITS_HPP