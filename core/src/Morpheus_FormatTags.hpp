/**
 * Morpheus_FormatTags.hpp
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

#ifndef MORPHEUS_FORMATTAGS_HPP
#define MORPHEUS_FORMATTAGS_HPP

#include <Morpheus_FormatTraits.hpp>

#include <impl/Morpheus_MatrixTags.hpp>
#include <impl/Morpheus_VectorTags.hpp>

namespace Morpheus {

/**
 * \addtogroup storage_format_tags Storage Format Tags
 * \brief Data structures used to tag data types
 * \ingroup wrappers_and_tags
 * \{
 *
 */
/**
 * @brief Tag used to mark containers as Matrix containers (Sparse) with
 * Coordinate (COO) Storage Format
 *
 */
struct CooFormatTag : public Impl::SparseMatrixTag {};
/**
 * @brief Tag used to mark containers as Matrix containers (Sparse) with
 * Compressed Sparse Row (CSR) Storage Format
 *
 */
struct CsrFormatTag : public Impl::SparseMatrixTag {};
/**
 * @brief Tag used to mark containers as Matrix containers (Sparse) with
 * Diagonal (DIA) Storage Format
 *
 */
struct DiaFormatTag : public Impl::SparseMatrixTag {};
/**
 * @brief Tag used to mark containers as Matrix container with
 * Dynamic Storage Format.
 *
 */
struct DynamicMatrixFormatTag : public Impl::DynamicMatrixTag {};

/**
 * @brief Tag used to mark containers as Matrix containers (Dense) with
 * Dense Format
 *
 */
struct DenseMatrixFormatTag : public Impl::DenseMatrixTag {};
/**
 * @brief Tag used to mark containers as Vector Containers (Dense) with
 * Dense Format
 *
 */
struct DenseVectorFormatTag : public Impl::DenseVectorTag {};

/*! \} // end of storage_format_tags group
 */

/**
 * \addtogroup format_traits Format Traits
 * \ingroup type_traits
 * \{
 *
 */
/**
 * @brief Checks if the given type \p T is a valid COO Sparse Matrix Format
 * Container i.e is valid matrix container and has \p CooFormatTag as \p tag
 * member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_coo_matrix_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_matrix_container<U>::value &&
          std::is_same<CooFormatTag, typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_coo_matrix_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_coo_matrix_format_container_v =
    is_coo_matrix_format_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid CSR Sparse Matrix Format
 * Container i.e is valid matrix container and has \p CsrFormatTag as \p tag
 * member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_csr_matrix_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_matrix_container_v<U> &&
          std::is_same<CsrFormatTag, typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_csr_matrix_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_csr_matrix_format_container_v =
    is_csr_matrix_format_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid DIA Sparse Matrix Format
 * Container i.e is valid matrix container and has \p DiaFormatTag as \p tag
 * member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dia_matrix_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_matrix_container_v<U> &&
          std::is_same<DiaFormatTag, typename U::tag>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dia_matrix_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dia_matrix_format_container_v =
    is_dia_matrix_format_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dynamic Matrix Format
 * Container i.e is valid matrix container and has \p DynamicMatrixFormatTag as
 * \p tag member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dynamic_matrix_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_matrix_container_v<U> &&
          std::is_same<DynamicMatrixFormatTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dynamic_matrix_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dynamic_matrix_format_container_v =
    is_dynamic_matrix_format_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dense Matrix Format
 * Container i.e is valid matrix container and has \p DenseMatrixFormatTag as \p
 * tag member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_matrix_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_matrix_container_v<U> &&
          std::is_same<DenseMatrixFormatTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_matrix_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_matrix_format_container_v =
    is_dense_matrix_format_container<T>::value;

/**
 * @brief Checks if the given type \p T is a valid Dense Vector Format
 * Container i.e is valid vector container and has \p DenseVectorFormatTag as \p
 * tag member trait.
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_dense_vector_format_container {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<
          is_vector_container<U>::value &&
          std::is_same<DenseVectorFormatTag, typename U::tag>::value>::type* =
          nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_dense_vector_format_container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_dense_vector_format_container_v =
    is_dense_vector_format_container<T>::value;

/*! \} // end of type_traits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_FORMATTAGS_HPP