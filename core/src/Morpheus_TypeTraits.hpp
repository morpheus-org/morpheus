/**
 * Morpheus_TypeTraits.hpp
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
#ifndef MORPHEUS_TYPETRAITS_HPP
#define MORPHEUS_TYPETRAITS_HPP

#include <Morpheus_FormatTags.hpp>

#include <fwd/Morpheus_Fwd_MatrixBase.hpp>
#include <fwd/Morpheus_Fwd_DenseMatrix.hpp>

#include <impl/Morpheus_Variant.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Morpheus {
// forward decl
template <typename Space>
struct KokkosSpace;

namespace Impl {
template <typename T>
struct is_kokkos_space_helper : std::false_type {};

template <typename Space>
struct is_kokkos_space_helper<KokkosSpace<Space>> : std::true_type {};

template <typename T, typename VariantContainer>
struct is_variant_member;

template <typename T, typename... Ts>
struct is_variant_member<T, Variant::variant<Ts...>>
    : public std::disjunction<std::is_same<T, Ts>...> {};

}  // namespace Impl

/**
 * @brief SFINAE Test to determine if the given type \p T is a member of \p
 * Variant
 *
 * @tparam T Type passed to check if is a member of the variant
 * @tparam Variant A variant container such as \p std::variant or
 * \p mpark::variant
 */
template <typename T, typename Variant>
inline constexpr bool is_variant_member_v =
    Impl::is_variant_member<T, Variant>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T has \p tag as a member
 * trait
 *
 * @tparam T Type passed to check if has \p tag as member trait
 */
template <class T>
class has_tag_trait {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(typename U::tag*);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Matrix
 * Container.
 *
 * @tparam T Type passed to check if is a valid Matrix Container.
 */
template <class T>
class is_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<
          std::is_base_of<Impl::MatrixTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_matrix_container SFINAE Test to check if the type
 * \p T is a valid Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Matrix Container.
 */
template <typename T>
inline constexpr bool is_matrix_container_v = is_matrix_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Sparse
 * Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Sparse Matrix Container.
 */
template <class T>
class is_sparse_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<
          std::is_base_of<Impl::SparseMatTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_sparse_matrix_container SFINAE Test to check if
 * the type \p T is a valid Sparse Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Sparse Matrix Container.
 */
template <typename T>
inline constexpr bool is_sparse_matrix_container_v =
    is_sparse_matrix_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Dense
 * Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Dense Matrix Container.
 */
template <class T>
class is_dense_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<
          std::is_base_of<Impl::DenseMatTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_matrix_container SFINAE Test to check if
 * the type \p T is a valid Dense Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Dense Matrix Container.
 */
template <typename T>
inline constexpr bool is_dense_matrix_container_v =
    is_dense_matrix_container<T>::value;

// template <template <class T, class... P> class Container, class T, class...
// P> struct is_sparse_matrix_class
//     : std::integral_constant<
//           bool, std::is_base_of<Impl::MatrixBase<Container, T, P...>,
//                                 Container<T, P...>>::value> {};

// template <template <class T, class... P> class Container, class T, class...
// P> struct is_dense_matrix_container
//     : std::integral_constant<
//           bool, std::is_same<DenseMatrix<T, P...>, Container<T,
//           P...>>::value> {
// };

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Vector
 * Container.
 *
 * @tparam T Type passed to check if is a valid Vector Container.
 */
template <class T>
class is_vector_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<
          std::is_base_of<Impl::VectorTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_vector_container SFINAE Test to check if the type
 * \p T is a valid Vector Container.
 *
 * @tparam T Type passed to check if is a valid Vector Container.
 */
template <typename T>
inline constexpr bool is_vector_container_v = is_vector_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Container.
 *
 * @tparam T Type passed to check if is a valid Vector Container.
 */
template <class T>
class is_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<is_matrix_container_v<U> ||
                              is_vector_container_v<U>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_vector_container SFINAE Test to check if the type
 * \p T is a valid Container.
 *
 * @tparam T Type passed to check if is a valid Container.
 */
template <typename T>
inline constexpr bool is_container_v = is_container<T>::value;

template <typename T1, typename T2>
struct is_same_format
    : std::integral_constant<
          bool, std::is_same<typename T1::tag, typename T2::tag>::value> {};

template <typename T1, typename T2>
struct is_compatible_container
    : std::integral_constant<bool,
                             std::is_same<typename T1::memory_space,
                                          typename T2::memory_space>::value &&
                                 std::is_same<typename T1::value_type,
                                              typename T2::value_type>::value &&
                                 std::is_same<typename T1::index_type,
                                              typename T2::index_type>::value> {
};

template <typename T1, typename T2>
struct is_compatible_type
    : std::integral_constant<
          bool, is_same_format<T1, T2>::value &&
                    std::is_same<typename T1::memory_space,
                                 typename T2::memory_space>::value &&
                    std::is_same<typename T1::value_type,
                                 typename T2::value_type>::value &&
                    std::is_same<typename T1::index_type,
                                 typename T2::index_type>::value> {};

template <typename T1, typename T2>
struct is_compatible_from_different_space
    : std::integral_constant<
          bool, is_same_format<T1, T2>::value &&
                    !std::is_same<typename T1::memory_space,
                                  typename T2::memory_space>::value &&
                    std::is_same<typename T1::value_type,
                                 typename T2::value_type>::value &&
                    std::is_same<typename T1::index_type,
                                 typename T2::index_type>::value> {};

template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class T>
struct is_arithmetic {
  static_assert(std::is_arithmetic<typename remove_cvref<T>::type>::value,
                "Morpheus: Invalid arithmetic type.");
  using container = is_arithmetic;
  using type      = T;
};

template <class ExecSpace>
inline constexpr bool is_execution_space_v =
    Kokkos::Impl::is_execution_space<ExecSpace>::value;

template <class MemorySpace>
inline constexpr bool is_Host_Memoryspace_v =
    std::is_same<typename Kokkos::HostSpace::memory_space, MemorySpace>::value;

template <class Space>
inline constexpr bool is_HostSpace_v =
    std::is_same<typename Kokkos::HostSpace,
                 typename Space::memory_space>::value;

template <class ExecSpace>
inline constexpr bool is_Serial_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::Serial::execution_space>::value;

#if defined(MORPHEUS_ENABLE_OPENMP)
template <class ExecSpace>
inline constexpr bool is_OpenMP_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::OpenMP::execution_space>::value;
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
template <class ExecSpace>
inline constexpr bool is_Cuda_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::Cuda::execution_space>::value;
#endif  // MORPHEUS_ENABLE_CUDA

// Takes arbitrary number of containers and checks if ExecSpace has access to
// all of them
template <typename ExecSpace, typename... Ts>
struct has_access;

template <typename ExecSpace, typename T, typename... Ts>
struct has_access<ExecSpace, T, Ts...> {
  static const bool value =
      Kokkos::Impl::SpaceAccessibility<ExecSpace,
                                       typename T::memory_space>::accessible &&
      has_access<ExecSpace, Ts...>::value;
};

template <typename ExecSpace, typename T>
struct has_access<ExecSpace, T> {
  static_assert(is_execution_space_v<ExecSpace>,
                "ExecSpace must be a valid execution space");
  static const bool value =
      Kokkos::Impl::SpaceAccessibility<ExecSpace,
                                       typename T::memory_space>::accessible;
};

template <class ExecSpace, class... T>
inline constexpr bool has_access_v = has_access<ExecSpace, T...>::value;

template <typename T, typename = void>
struct has_kokkos_space : std::false_type {};

template <typename T>
struct has_kokkos_space<T, std::void_t<typename T::kokkos_space>>
    : std::true_type {};

template <typename T>
using is_kokkos_space = typename Impl::is_kokkos_space_helper<
    typename std::remove_cv<T>::type>::type;

template <class T>
inline constexpr bool is_kokkos_space_v = is_kokkos_space<T>::value;

}  // namespace Morpheus

#endif  // MORPHEUS_TYPETRAITS_HPP