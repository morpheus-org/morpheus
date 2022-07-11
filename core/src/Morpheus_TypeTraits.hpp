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
 * \addtogroup utilities Utilities
 * \par Overview
 * TODO
 *
 */

/**
 * \addtogroup typetraits Type Traits
 * \brief Various tools for examining the different types available and
 * relationships between them during compile-time.
 * \ingroup utilities
 * \{
 *
 */

template <typename T>
struct IndexType {
  static_assert(std::is_integral<T>::value, "T needs to be an integral type!");
  using index_type = T;
  using type       = IndexType;
};

template <typename T>
struct ValueType {
  static_assert(std::is_scalar<T>::value, "T needs to be a scalar type!");
  using value_type = T;
  using type       = ValueType;
};

/**
 * @brief Checks if the given type \p T is a member of \p Variant container such
 * as \p std::variant or \p mpark::variant
 *
 * @tparam T Type passed for check
 * @tparam Variant A variant container
 *
 */
template <typename T, typename Variant>
inline constexpr bool is_variant_member_v =
    Impl::is_variant_member<T, Variant>::value;

/**
 * @brief Checks if \p T has \p tag as a member trait.
 *
 * @tparam T Type passed for check
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
 * @brief Short-hand for \p has_tag_trait.
 *
 * @tparam T Type passed for check
 */
template <typename T>
inline constexpr bool has_tag_trait_v = has_tag_trait<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Matrix Container i.e has a
 * \p tag member trait that is a derived class of \p MatrixTag.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_matrix_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_matrix_container_v = is_matrix_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Sparse Matrix Container i.e
 * has a \p tag member trait that is a derived class of \p SparseMatTag.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_sparse_matrix_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_sparse_matrix_container_v =
    is_sparse_matrix_container<T>::value;

/**
 * @brief Check if the given type \p T is a valid Dense Matrix Container i.e
 * has a \p tag member trait that is a derived class of \p DenseMatTag.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_dense_matrix_container.
 *
 * @tparam T Type passed for checks.
 */
template <typename T>
inline constexpr bool is_dense_matrix_container_v =
    is_dense_matrix_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Dynamic
 * Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Dynamic Matrix Container.
 */
template <class T>
class is_dynamic_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(typename U::tag*,
                   typename std::enable_if<std::is_base_of<
                       DynamicTag, typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dynamic_matrix_container SFINAE Test to check if
 * the type \p T is a valid Dynamic Matrix Container.
 *
 * @tparam T Type passed to check if is a valid Dynamic Matrix Container.
 */
template <typename T>
inline constexpr bool is_dynamic_matrix_container_v =
    is_dynamic_matrix_container<T>::value;

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
 * @tparam T Type passed to check if is a valid Container.
 */
template <typename T>
inline constexpr bool is_vector_container_v = is_vector_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T is a valid Container.
 *
 * @tparam T Type passed to check if is a valid Container.
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
 * @brief Short-hand to \p is_container SFINAE Test to check if the type
 * \p T is a valid Container.
 *
 * @tparam T Type passed to check if is a valid Container.
 */
template <typename T>
inline constexpr bool is_container_v = is_container<T>::value;

/**
 * @brief SFINAE Test to determine if the given type \p T1 is the same format as
 * type \p T2 - Two containers have the same format if they hold the same tag.
 *
 * @tparam T1 Type passed to check if has the same format as \p T2
 * @tparam T2 Reference type against which we compare.
 */
template <class T1, class T2>
class is_same_format {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::tag*, typename U2::tag*,
      typename std::enable_if<
          is_container_v<U1> && is_container_v<U2> &&
          std::is_same<typename U1::tag, typename U2::tag>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_same_format SFINAE Test to check if the given type
 * \p T1 is the same format as type \p T2.
 *
 * @tparam T1 Type passed to check if has the same format as \p T2
 * @tparam T2 Reference type against which we compare.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_format_v = is_same_format<T1, T2>::value;

/**
 * @brief SFINAE Test to determine if the two types are in the same memory space
 *
 * @tparam T1 Type passed to check if is in the same memory space as \p T2
 * @tparam T2 Type passed to check if is in the same memory space as \p T1
 */
template <class T1, class T2>
class in_same_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::memory_space*, typename U2::memory_space*,
      typename std::enable_if<std::is_same<
          typename U1::memory_space, typename U2::memory_space>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p in_same_memory_space SFINAE Test to check if the
 * given types are in the same memory space.
 *
 * @tparam T1 Type passed to check if is in the same memory space as \p T2
 * @tparam T2 Type passed to check if is in the same memory space as \p T1
 */
template <typename T1, typename T2>
inline constexpr bool in_same_memory_space_v =
    in_same_memory_space<T1, T2>::value;

/**
 * @brief SFINAE Test to determine if the two types have the same layout
 *
 * @tparam T1 Type passed to check if has the same layout as \p T2
 * @tparam T2 Type passed to check if has the same layout as \p T1
 */
template <class T1, class T2>
class have_same_layout {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::array_layout*, typename U2::array_layout*,
      typename std::enable_if<std::is_same<
          typename U1::array_layout, typename U2::array_layout>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p have_same_layout SFINAE Test to check if the
 * given types have the same layout.
 *
 * @tparam T1 Type passed to check if has the same layout as \p T2
 * @tparam T2 Type passed to check if has the same layout as \p T1
 */
template <typename T1, typename T2>
inline constexpr bool have_same_layout_v = have_same_layout<T1, T2>::value;

/**
 * @brief SFINAE Test to determine if the two types hold the same value type
 *
 * @tparam T1 Type passed to check if holds the same value type as \p T2
 * @tparam T2 Type passed to check if holds the same value type as \p T1
 */
template <class T1, class T2>
class have_same_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::value_type*, typename U2::value_type*,
      typename std::enable_if<std::is_same<
          typename U1::value_type, typename U2::value_type>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p have_same_value_type SFINAE Test to check if the
 * given types hold the same value type.
 *
 * @tparam T1 Type passed to check if holds the same value type as \p T2
 * @tparam T2 Type passed to check if holds the same value type as \p T1
 */
template <typename T1, typename T2>
inline constexpr bool have_same_value_type_v =
    have_same_value_type<T1, T2>::value;

/**
 * @brief SFINAE Test to determine if the two types hold the same index type
 *
 * @tparam T1 Type passed to check if holds the same index type as \p T2
 * @tparam T2 Type passed to check if holds the same index type as \p T1
 */
template <class T1, class T2>
class have_same_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::index_type*, typename U2::index_type*,
      typename std::enable_if<std::is_same<
          typename U1::index_type, typename U2::index_type>::value>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p have_same_index_type SFINAE Test to check if the
 * given types hold the same index type.
 *
 * @tparam T1 Type passed to check if holds the same index type as \p T2
 * @tparam T2 Type passed to check if holds the same index type as \p T1
 */
template <typename T1, typename T2>
inline constexpr bool have_same_index_type_v =
    have_same_index_type<T1, T2>::value;

/**
 * @brief SFINAE to determine if the two types are compatible i.e in the same
 * memory space, holding the same type of value and indices and with the same
 * memory layout.
 *
 * @tparam T1 First container to compare
 * @tparam T2 Second container to compare
 */
template <class T1, class T2>
class is_compatible {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::tag*, typename U2::tag*,
      typename std::enable_if<in_same_memory_space_v<U1, U2> &&
                              have_same_layout_v<U1, U2> &&
                              have_same_value_type_v<U1, U2> &&
                              have_same_index_type_v<U1, U2>>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_compatible SFINAE Test to check if the
 * given types are compatible types.
 *
 * @tparam T1 First container to compare
 * @tparam T2 Second container to compare
 */
template <typename T1, typename T2>
inline constexpr bool is_compatible_v = is_compatible<T1, T2>::value;

/**
 * @brief SFINAE to determine if the two types are dynamically compatible i.e
 * compatible and at least one of them is a dynamic container.
 *
 * @tparam T1 First container to compare
 * @tparam T2 Second container to compare
 */
template <class T1, class T2>
class is_dynamically_compatible {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      typename U1::tag*, typename U2::tag*,
      typename std::enable_if<(is_dynamic_matrix_container<U1>::value ||
                               is_dynamic_matrix_container<U2>::value) &&
                              is_compatible_v<U1, U2>>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dynamically_compatible SFINAE Test to check if the
 * given types are dynamically compatible types.
 *
 * @tparam T1 First container to compare
 * @tparam T2 Second container to compare
 */
template <typename T1, typename T2>
inline constexpr bool is_dynamically_compatible_v =
    is_dynamically_compatible<T1, T2>::value;

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
/*! \}
 */
}  // namespace Morpheus

#endif  // MORPHEUS_TYPETRAITS_HPP