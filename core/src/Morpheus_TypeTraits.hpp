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

#include <impl/Morpheus_Variant.hpp>

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
 * \defgroup type_traits Type Traits
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

/*! \} end of type_traits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_TYPETRAITS_HPP