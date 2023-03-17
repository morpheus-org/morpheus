/**
 * Morpheus_FormatTraits.hpp
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
#ifndef MORPHEUS_FORMAT_TRAITS_HPP
#define MORPHEUS_FORMAT_TRAITS_HPP

// #include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_SpaceTraits.hpp>

#include <impl/Morpheus_MatrixTags.hpp>
#include <impl/Morpheus_VectorTags.hpp>

#include <type_traits>

namespace Morpheus {
/**
 * \defgroup format_traits Format Traits
 * \brief Various tools for examining the different types of containers
 * available and relationships between them during compile-time.
 * \ingroup type_traits
 *
 */

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

/*! \} end of format_traits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_FORMAT_TRAITS_HPP