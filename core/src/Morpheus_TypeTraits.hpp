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
#include <impl/Morpheus_MatrixTags.hpp>
#include <impl/Morpheus_VectorTags.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Morpheus {

/**
 * \defgroup utilities Utilities
 * \par Overview
 * TODO
 *
 */
/**
 * \defgroup typetraits Type Traits
 * \brief Various tools for examining the different types available and
 * relationships between them during compile-time.
 * \ingroup utilities
 *
 */

/*! \cond */
namespace Impl {

template <typename T, typename VariantContainer>
struct is_variant_member;

template <typename T, typename... Ts>
struct is_variant_member<T, Variant::variant<Ts...>>
    : public std::disjunction<std::is_same<T, Ts>...> {};

}  // namespace Impl
/*! \endcond */

/**
 * \addtogroup wrappers Wrappers
 * \brief Data structures used to wrap around data types
 * \ingroup wrappers_and_tags
 * \{
 */
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
/*! \} // end of wrappers group
 */

/**
 * \addtogroup typetraits Type Traits
 * \ingroup utilities
 * \{
 */

/**
 * @brief Checks if the given type \p T is a member of \p Variant container such
 * as \p std::variant or \p mpark::variant
 *
 * @tparam T Type passed for check
 * @tparam Variant A variant container
 *
 */
template <typename T, typename Variant>
class is_variant_member {
  typedef char yes[1];
  typedef char no[2];

  template <typename U, typename UV>
  static yes& test(typename U::type*, UV*,
                   typename std::enable_if<Impl::is_variant_member<
                       typename U::type, UV>::value>::type* = nullptr);

  template <typename U, typename UV>
  static yes& test(
      U*, UV*,
      typename std::enable_if<Impl::is_variant_member<U, UV>::value>::type* =
          nullptr);

  template <typename U, typename UV>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T, Variant>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand for \p is_variant_member.
 *
 * @tparam T Type passed for check
 */
template <typename T, typename Variant>
inline constexpr bool is_variant_member_v =
    is_variant_member<T, Variant>::value;

#define MORPHEUS_IMPL_HAS_TRAIT(TRAIT)                                 \
  template <class T>                                                   \
  class has_##TRAIT {                                                  \
    typedef char yes[1];                                               \
    typedef char no[2];                                                \
                                                                       \
    template <class U>                                                 \
    static yes& test(typename U::TRAIT*);                              \
                                                                       \
    template <class U>                                                 \
    static no& test(...);                                              \
                                                                       \
   public:                                                             \
    static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes); \
  };

/**
 * @brief Checks if \p T has \p tag as a member trait.
 *
 * @tparam T Type passed for check
 * @tparam Variant A variant container
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
 * @brief A valid Matrix container is the one that has a valid Matrix tag i.e
 * satisfies the \p has_matrix_tag check. Note that both dense and sparse
 * matrices should be valid matrix containers.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<Impl::has_matrix_tag<U>::value>::type* = nullptr);

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
 * @brief A valid Sparse Matrix container is the one that has a valid Sparse
 * Matrix tag i.e satisfies the \p has_sparse_matrix_tag check. Note that any
 * supported sparse matrix storage format should be a valid Sparse Matrix
 * Container.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_sparse_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<Impl::has_sparse_matrix_tag_v<U>>::type* =
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
 * @brief A valid Dense Matrix container is the one that has a valid Dense
 * Matrix tag i.e satisfies the \p has_dense_matrix_tag check.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<Impl::has_dense_matrix_tag_v<U>>::type* =
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
 * @brief A valid Vector container is the one that has a valid Vector tag i.e
 * satisfies the \p has_vector_tag check. Note that a Vector container could be
 * either dense or sparse.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_vector_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<Impl::has_vector_tag<U>::value>::type* = nullptr);

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
 * @brief Checks if the given type \p T is a valid Morpheus Container i.e
 * is either a valid matrix or a vector container that satisfies
 * \p is_matrix_container or \p is_vector_container.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<is_matrix_container_v<U> ||
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
 * @brief A valid Dynamic Matrix container is the one that has a valid Dynamic
 * Matrix tag i.e satisfies the \p has_dynamic_matrix_tag check. Note that any
 * supported dynamic matrix storage format should be a valid Dynamic Matrix
 * Container.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dynamic_matrix_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<Impl::has_dynamic_matrix_tag<U>::value>::type* =
          nullptr);

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
class has_same_format {
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
 * @brief Short-hand to \p has_same_format.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool has_same_format_v = has_same_format<T1, T2>::value;

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
      U*, typename std::enable_if<
              (std::is_same<Kokkos::LayoutLeft, U>::value ||
               std::is_same<Kokkos::LayoutRight, U>::value)>::type* = nullptr);

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
 * @brief Checks if the given type \p T has a layout i.e has as a
 * \p array_layout member trait it self and is one of the supported layouts.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_layout {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_layout_v<typename U::array_layout>>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_layout.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_layout_v = has_layout<T>::value;

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
      typename std::enable_if<is_layout_v<U1> && is_layout_v<U2> &&
                              std::is_same<U1, U2>::value>::type* = nullptr);

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
 * @brief Checks if the two types have the same valid supported layout
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class has_same_layout {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<is_same_layout_v<
          typename U1::array_layout, typename U2::array_layout>>::type* =
          nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_same_layout.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool has_same_layout_v = has_same_layout<T1, T2>::value;

/**
 * @brief Checks if the given type \p T is a valid value type i.e a scalar
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_scalar<U>::value>::type* = nullptr);

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
 * @brief Checks if the given type \p T has a valid value type i.e a scalar
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_value_type_v<typename U::value_type>>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_value_type.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_value_type_v = has_value_type<T>::value;

/**
 * @brief Checks if the two types are of type value_type and the same.
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
      typename std::enable_if<is_value_type_v<U1> && is_value_type_v<U2> &&
                              std::is_same<U1, U2>::value>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_same_value_type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool is_same_value_type_v = is_same_value_type<T1, T2>::value;

/**
 * @brief Checks if the two types have the same valid value type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class has_same_value_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<is_same_value_type_v<
          typename U1::value_type, typename U2::value_type>>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_same_value_type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool has_same_value_type_v =
    has_same_value_type<T1, T2>::value;

/**
 * @brief Checks if the given type \p T is a valid index type i.e an integral
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_integral<U>::value>::type* = nullptr);

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
 * @brief Checks if the given type \p T has a valid index type i.e an integral
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_index_type_v<typename U::index_type>>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_index_type.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_index_type_v = has_index_type<T>::value;

/**
 * @brief Checks if the two types is of type index_type and are the same.
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
      typename std::enable_if<is_index_type_v<U1> && is_index_type_v<U2> &&
                              std::is_same<U1, U2>::value>::type* = nullptr);

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
 * @brief Checks if the two types have the same valid index type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class has_same_index_type {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<is_same_index_type_v<
          typename U1::index_type, typename U2::index_type>>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_same_index_type.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <typename T1, typename T2>
inline constexpr bool has_same_index_type_v =
    has_same_index_type<T1, T2>::value;

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
      typename std::enable_if<has_same_memory_space_v<U1, U2> &&
                              has_same_layout_v<U1, U2> &&
                              has_same_value_type_v<U1, U2> &&
                              has_same_index_type_v<U1, U2>>::type* = nullptr);

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
    : std::integral_constant<bool, has_same_format_v<T1, T2> &&
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

/**
 * @brief Checks if the two types are format compatible containers but from
 * different memory space i.e have the same storage format and are compatible
 * containers with relaxed memory space and layout requirements.
 *
 * @tparam T1 First type passed for comparison.
 * @tparam T2 Second type passed for comparison.
 */
template <class T1, class T2>
class is_format_compatible_different_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U1, class U2>
  static yes& test(
      U1*, U2*,
      typename std::enable_if<
          !has_same_memory_space_v<U1, U2> && has_same_format_v<U1, U2> &&
          has_same_layout_v<U1, U2> && has_same_value_type_v<U1, U2> &&
          has_same_index_type_v<U1, U2>>::type* = nullptr);

  template <class U1, class U2>
  static no& test(...);

 public:
  static const bool value =
      sizeof(test<T1, T2>(nullptr, nullptr)) == sizeof(yes);
};

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

/*! \} end of typetraits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_TYPETRAITS_HPP