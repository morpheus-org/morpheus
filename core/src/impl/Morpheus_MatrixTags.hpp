/**
 * Morpheus_MatrixTags.hpp
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

#ifndef MORPHEUS_MATRIXTAGS_HPP
#define MORPHEUS_MATRIXTAGS_HPP

#include <type_traits>

namespace Morpheus {
namespace Impl {
/**
 * @brief Tag used to mark containers as Matrices
 *
 */
struct MatrixTag {};
/**
 * @brief Tag used to mark containers as Sparse Matrices
 *
 */
struct SparseMatrixTag : public MatrixTag {};
/**
 * @brief Tag used to mark containers as Dense Matrices
 *
 */
struct DenseMatrixTag : public MatrixTag {};

/**
 * @brief Checks if the given type \p T is a valid Matrix Tag i.e is a derived
 * class of \p MatrixTag
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_base_of<MatrixTag, U>::value>::type* =
              nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_matrix_tag_v = is_matrix_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a tag trait of type \p MatrixTag
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<is_matrix_tag_v<typename U::tag>>::type* =
              nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_matrix_tag_v = has_matrix_tag<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Sparse Matrix Tag i.e
 * is a derived class of \p SparseMatrixTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_sparse_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_base_of<SparseMatrixTag, U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_sparse_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_sparse_matrix_tag_v = is_sparse_matrix_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a valid Sparse Matrix Tag i.e
 * has a \p tag member trait that is a derived class of \p SparseMatrixTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_sparse_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_sparse_matrix_tag_v<typename U::tag>>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_sparse_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_sparse_matrix_tag_v = has_sparse_matrix_tag<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dense Matrix Container i.e
 * it is a derived class of \p DenseMatrixTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_base_of<DenseMatrixTag, U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_matrix_tag_v = is_dense_matrix_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a valid Dense Matrix Tag i.e
 * has a \p tag member trait that is a derived class of \p DenseMatrixTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_dense_matrix_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_dense_matrix_tag_v<typename U::tag>>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_dense_matrix_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_dense_matrix_tag_v = has_dense_matrix_tag<T>::value;

}  // namespace Impl

// Matrix Format Tag Wrapper
template <class T>
struct MatrixFormatTag {
  static_assert(std::is_base_of<Impl::MatrixTag, T>::value,
                "Morpheus: Invalid Matrix Format tag.");
  using format_tag = MatrixFormatTag;
  using tag        = T;
};

}  // namespace Morpheus
#endif  // MORPHEUS_MATRIXTAGS_HPP