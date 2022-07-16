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

/*! \cond */
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
/*! \endcond */

/**
 * @brief A wrapper that checks if the provided type is an integral type.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
struct IndexType {
  static_assert(std::is_integral<T>::value, "T needs to be an integral type!");
  using index_type = T;
  using type       = IndexType;
};

/**
 * @brief A wrapper that checks if the provided type is a scalar type.
 *
 * @tparam T Type passed for check.
 */
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
 * @brief Checks if the given type \p T is a valid Dense Matrix Container i.e
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
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_matrix_container_v =
    is_dense_matrix_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Vector Container i.e
 * has a \p tag member trait that is a derived class of \p VectorTag.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_vector_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_vector_container_v = is_vector_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dense Vector Container i.e
 * has a \p tag member trait that is a derived class of \p DenseVectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_vector_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      typename U::tag*,
      typename std::enable_if<
          std::is_base_of<DenseVectorTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_vector_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_vector_container_v =
    is_dense_vector_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Morpheus Container i.e
 * has a \p tag member trait and is either a Morpheus Matrix or Vector
 * container.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_container_v = is_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dynamic Matrix Container i.e
 * has a \p tag member trait and is a derived class of \p DynamicTag.
 *
 * @tparam T Type passed for check.
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
 * @brief Short-hand to \p is_dynamic_matrix_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dynamic_matrix_container_v =
    is_dynamic_matrix_container<T>::value;

/**
 * @brief Checks if the two types have the same format i.e both are valid
 * containers and have the same \p tag member trait.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
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
 * @brief Short-hand to \p is_same_format.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_format_v = is_same_format<T1, T2>::value;

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
      U*,
      typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
          std::is_same<typename U::memory_space, Kokkos::HostSpace>::value ||
#endif
#if defined(MORPHEUS_ENABLE_CUDA)
          std::is_same<typename U::memory_space, Kokkos::CudaSpace>::value ||
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
      typename std::enable_if<
          is_memory_space<U1>::value && is_memory_space<U2>::value &&
          std::is_same<typename U1::memory_space,
                       typename U2::memory_space>::value>::type* = nullptr);

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
 * @brief Checks if the given type \p T is a layout i.e has as a
 * \p array_layout member trait it self and is one of the supported layouts.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_layout {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          (std::is_same<Kokkos::LayoutLeft, typename U::array_layout>::value ||
           std::is_same<Kokkos::LayoutRight,
                        typename U::array_layout>::value)>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_layout.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_layout_v = is_layout<T>::value;

/**
 * @brief Checks if the two types have the same valid supported layout
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_same_layout {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          is_layout_v<U1> && is_layout_v<U2> &&
          std::is_same<typename U1::array_layout,
                       typename U2::array_layout>::value>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_same_layout.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_layout_v = is_same_layout<T1, T2>::value;

/**
 * @brief Checks if the given type \p T has a valid value type i.e a scalar
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_scalar<typename U::value_type>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_value_type.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_value_type_v = is_value_type<T>::value;

/**
 * @brief Checks if the two types have the same valid value type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_same_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          is_value_type_v<U1> && is_value_type_v<U2> &&
          std::is_same<typename U1::value_type,
                       typename U2::value_type>::value>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_value_type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_value_type_v = is_same_value_type<T1, T2>::value;

/**
 * @brief Checks if the given type \p T has a valid index type i.e an integral
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          std::is_integral<typename U::index_type>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_index_type.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_index_type_v = is_index_type<T>::value;

/**
 * @brief Checks if the two types have the same valid index type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_same_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          is_index_type_v<U1> && is_index_type_v<U2> &&
          std::is_same<typename U1::index_type,
                       typename U2::index_type>::value>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_same_index_type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_index_type_v = is_same_index_type<T1, T2>::value;

/**
 * @brief Checks if the two types are compatible containers i.e are in the same
 * memory space and have the same layout, index and value type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_compatible {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          is_same_memory_space_v<U1, U2> && is_same_layout_v<U1, U2> &&
          is_same_value_type_v<U1, U2> && is_same_index_type_v<U1, U2>>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_compatible.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_compatible_v = is_compatible<T1, T2>::value;

/**
 * @brief Checks if the two types are dynamically compatible containers i.e are
 * compatible containers and at least one of them is also a dynamic container.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
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
 * @brief Short-hand to \p is_dynamically_compatible.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_dynamically_compatible_v =
    is_dynamically_compatible<T1, T2>::value;

/**
 * @brief Checks if the two types are format compatible containers i.e are
 * compatible containers and have the same storage format.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
struct is_format_compatible
    : std::integral_constant<bool, is_same_format_v<T1, T2> &&
                                       is_compatible_v<T1, T2>> {};

/**
 * @brief Short-hand to \p is_format_compatible.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_format_compatible_v =
    is_format_compatible<T1, T2>::value;

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

/**
 * @brief Provides the member type which is the same as T, except that its
 * topmost const- and reference-qualifiers are removed
 *
 * @tparam T Type passed for conversion.
 */
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

/**
 * @brief Short-hand to \p remove_cvref.
 *
 * @tparam T Type passed for conversion.
 */
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

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
      U*,
      typename std::enable_if<
          std::is_same<typename U::execution_space,
                       Kokkos::DefaultHostExecutionSpace>::value ||
          std::is_same<typename U::execution_space,
                       Kokkos::DefaultExecutionSpace>::value ||
#if defined(MORPHEUS_ENABLE_SERIAL)
          std::is_same<typename U::execution_space, Kokkos::Serial>::value ||
#endif
#if defined(MORPHEUS_ENABLE_OPENMP)
          std::is_same<typename U::execution_space, Kokkos::OpenMP>::value ||
#endif
#if defined(MORPHEUS_ENABLE_CUDA)
          std::is_same<typename U::execution_space, Kokkos::Cuda>::value ||
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
 * @brief Checks if the given type \p T is a valid supported Host memory space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_host_memory_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
          std::is_same<typename U::memory_space, Kokkos::HostSpace>::value ||
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
      U*,
      typename std::enable_if<
          std::is_same<typename U::execution_space,
                       Kokkos::DefaultHostExecutionSpace>::value ||
#if defined(MORPHEUS_ENABLE_SERIAL)
          std::is_same<typename U::execution_space, Kokkos::Serial>::value ||
#endif
#if defined(MORPHEUS_ENABLE_OPENMP)
          std::is_same<typename U::execution_space, Kokkos::OpenMP>::value ||
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
 * @brief Checks if the given type \p T is a Serial execution space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_serial_execution_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
#if defined(MORPHEUS_ENABLE_SERIAL)
          std::is_same<typename U::execution_space, Kokkos::Serial>::value ||
#endif
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
  static yes& test(
      U*,
      typename std::enable_if<
#if defined(MORPHEUS_ENABLE_OPENMP)
          std::is_same<typename U::execution_space, Kokkos::OpenMP>::value ||
#endif
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