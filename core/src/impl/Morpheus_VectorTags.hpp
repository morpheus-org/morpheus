/**
 * Morpheus_VectorTags.hpp
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

#ifndef MORPHEUS_VECTORTAGS_HPP
#define MORPHEUS_VECTORTAGS_HPP

#include <type_traits>

namespace Morpheus {
namespace Impl {

/**
 * @brief Tag used to mark containers as Vectors
 *
 */
struct VectorTag {};
/**
 * @brief Tag used to mark containers as Sparse Vectors
 *
 */
struct SparseVectorTag : public VectorTag {};
/**
 * @brief Tag used to mark containers as Dense Vectors
 *
 */
struct DenseVectorTag : public VectorTag {};

/**
 * @brief Checks if the given type \p T is a valid Vector Tag i.e
 * it is a derived class of \p VectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_base_of<VectorTag, U>::value>::type* =
              nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_vector_tag_v = is_vector_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a valid Vector Tag i.e
 * has a \p tag member trait that is a derived class of \p VectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<is_vector_tag<typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_vector_tag_v = has_vector_tag<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Sparse Vector Tag i.e
 * it is a derived class of \p SparseVectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_sparse_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_base_of<SparseVectorTag, U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_sparse_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_sparse_vector_tag_v = is_sparse_vector_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a valid Sparse Vector Tag i.e
 * has a \p tag member trait that is a derived class of \p SparseVectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_sparse_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              is_sparse_vector_tag<typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_sparse_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_sparse_vector_tag_v = has_sparse_vector_tag<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dense Vector Tag i.e
 * it is a derived class of \p DenseVectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              std::is_base_of<DenseVectorTag, U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_vector_tag_v = is_dense_vector_tag<T>::value;

/**
 * @brief Checks if the given type \p T has a valid Dense Vector Tag i.e
 * has a \p tag member trait that is a derived class of \p DenseVectorTag.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class has_dense_vector_tag {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              is_dense_vector_tag<typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_dense_vector_tag.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_dense_vector_tag_v = has_dense_vector_tag<T>::value;
}  // namespace Impl

// Vector Format Tag Wrapper
template <class T>
struct VectorFormatTag {
  static_assert(std::is_base_of<Impl::VectorTag, T>::value,
                "Morpheus: Invalid Vector Format tag.");
  using format_tag = VectorFormatTag;
  using tag        = T;
};

}  // namespace Morpheus

#endif  // MORPHEUS_VECTORTAGS_HPP