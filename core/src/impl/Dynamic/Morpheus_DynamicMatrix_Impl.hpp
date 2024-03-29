/**
 * Morpheus_DynamicMatrix_Impl.hpp
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

#ifndef MORPHEUS_DYNAMICMATRIX_IMPL_HPP
#define MORPHEUS_DYNAMICMATRIX_IMPL_HPP

#include <string>

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatsRegistry.hpp>
#include <Morpheus_ContainerTraits.hpp>

namespace Morpheus {

namespace Impl {

template <class ValueType, class... Properties>
struct any_type_resize
    : public Morpheus::ContainerTraits<any_type_resize, ValueType,
                                       Properties...> {
  using traits =
      Morpheus::ContainerTraits<any_type_resize, ValueType, Properties...>;
  using size_type   = typename traits::size_type;
  using value_type  = typename traits::value_type;
  using result_type = void;

  // Specialization for Coo resize with three arguments
  template <typename... Args>
  result_type operator()(
      typename CooMatrix<ValueType, Properties...>::type &mat,
      const size_type nrows, const size_type ncols, const size_type nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Csr resize with three arguments
  template <typename... Args>
  result_type operator()(
      typename CsrMatrix<ValueType, Properties...>::type &mat,
      const size_type nrows, const size_type ncols, const size_type nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Dia resize with five arguments
  template <typename... Args>
  result_type operator()(
      typename DiaMatrix<ValueType, Properties...>::type &mat,
      const size_type nrows, const size_type ncols, const size_type nnnz,
      const size_type ndiag, const size_type alignment = 32) {
    mat.resize(nrows, ncols, nnnz, ndiag, alignment);
  }

  // Specialization for Ell resize with five arguments
  template <typename... Args>
  result_type operator()(
      typename EllMatrix<ValueType, Properties...>::type &mat,
      const size_type nrows, const size_type ncols, const size_type nnnz,
      const size_type num_entries_per_row, const size_type alignment = 32) {
    mat.resize(nrows, ncols, nnnz, num_entries_per_row, alignment);
  }

  // Constrains any other overloads for supporting formats
  // Unsupported formats won't compile
  template <typename... Args>
  result_type operator()(
      typename CooMatrix<ValueType, Properties...>::type &mat, Args &&...) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic resize interface for current format (" +
        std::to_string(mat.format_index()) + ").");
  }

  template <typename... Args>
  result_type operator()(
      typename CsrMatrix<ValueType, Properties...>::type &mat, Args &&...) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic resize interface for current format (" +
        std::to_string(mat.format_index()) + ").");
  }

  template <typename... Args>
  result_type operator()(
      typename DiaMatrix<ValueType, Properties...>::type &mat, Args &&...) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic resize interface for current format (" +
        std::to_string(mat.format_index()) + ").");
  }

  template <typename... Args>
  result_type operator()(
      typename EllMatrix<ValueType, Properties...>::type &mat, Args &&...) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic resize interface for current format (" +
        std::to_string(mat.format_index()) + ").");
  }
};

struct any_type_resize_from_mat {
  using result_type = void;

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<has_same_format<T1, T2>::value>::type * =
          nullptr) {
    dst.resize(src);
  }

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<!has_same_format<T1, T2>::value>::type * =
          nullptr) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic resize interface. Src and dst tags must be "
        "the same (" +
        std::to_string(src.format_index()) +
        " != " + std::to_string(dst.format_index()) + ")");
  }
};

struct any_type_allocate {
  using result_type = void;

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<has_same_format<T1, T2>::value>::type * =
          nullptr) {
    dst = T2().allocate(src);
  }

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<!has_same_format<T1, T2>::value>::type * =
          nullptr) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic allocate interface. Src and std tags must "
        "be the same (" +
        std::to_string(src.format_index()) +
        " != " + std::to_string(dst.format_index()) + ")");
  }
};

struct any_type_assign {
  using result_type = void;

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<has_same_format<T1, T2>::value>::type * =
          nullptr) {
    dst = src;
  }

  template <typename T1, typename T2>
  result_type operator()(
      const T1 &src, T2 &dst,
      typename std::enable_if<!has_same_format<T1, T2>::value>::type * =
          nullptr) {
    throw Morpheus::RuntimeException(
        "Invalid use of the dynamic assign interface. Src and dst tags must be "
        "the same (" +
        std::to_string(src.format_index()) +
        " != " + std::to_string(dst.format_index()) + ")");
  }
};

template <size_t I, class ValueType, typename... Properties>
struct activate_impl {
  using variant   = typename MatrixFormats<ValueType, Properties...>::variant;
  using type_list = typename MatrixFormats<ValueType, Properties...>::type_list;

  static void activate(variant &A, size_t idx) {
    if (idx == I - 1) {
      A = typename type_list::template type<I - 1>{};
    } else {
      activate_impl<I - 1, ValueType, Properties...>::activate(A, idx);
    }
  }
};

// Base case, activate to the first type in the variant
template <class ValueType, typename... Properties>
struct activate_impl<0, ValueType, Properties...> {
  using variant   = typename MatrixFormats<ValueType, Properties...>::variant;
  using type_list = typename MatrixFormats<ValueType, Properties...>::type_list;

  static void activate(variant &A, size_t idx) {
    idx = 0;
    activate_impl<1, ValueType, Properties...>::activate(A, idx);
  }
};

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMICMATRIX_IMPL_HPP