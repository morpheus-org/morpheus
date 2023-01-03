/**
 * Morpheus_MatrixOperations_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_MatrixOperations_Impl.hpp>
#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
namespace Impl {
template <typename ExecSpace, typename Matrix, typename Vector>
inline void update_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_dense_vector_format_container<Vector>::value>::type* =
        nullptr) {
  Impl::Variant::visit(
      [&](auto&& arg) { Impl::update_diagonal<ExecSpace>(arg, diagonal); },
      A.formats());
}

template <typename ExecSpace, typename Matrix, typename Vector>
inline void get_diagonal(
    const Matrix& A, Vector& diagonal,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_dense_vector_format_container<Vector>::value>::type* =
        nullptr) {
  Impl::Variant::visit(
      [&](auto&& arg) { Impl::get_diagonal<ExecSpace>(arg, diagonal); },
      A.const_formats());
}

template <typename ExecSpace, typename Matrix, typename SizeType,
          typename ValueType>
inline void set_value(
    Matrix& A, SizeType row, SizeType col, ValueType value,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value>::type* =
        nullptr) {
  Impl::Variant::visit(
      [&](auto&& arg) { Impl::set_value<ExecSpace>(arg, row, col, value); },
      A.formats());
}

template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
inline void set_values(
    Matrix& A, typename IndexVector::value_type m, const IndexVector idxm,
    typename IndexVector::value_type n, const IndexVector idxn,
    ValueVector values,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_dense_vector_format_container<IndexVector>::value &&
        Morpheus::is_dense_vector_format_container<ValueVector>::value>::type* =
        nullptr) {
  Impl::Variant::visit(
      [&](auto&& arg) {
        Impl::set_value<ExecSpace>(arg, m, idxm, n, idxn, values);
      },
      A.formats());
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
inline void transpose(
    const Matrix& A, TransposeMatrix& At,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_sparse_matrix_container<TransposeMatrix>::value>::type* =
        nullptr) {
  Impl::Variant::visit([&](auto&& arg) { Impl::transpose<ExecSpace>(arg, At); },
                       A.const_formats());
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
inline void transpose(
    const Matrix& A, TransposeMatrix& At,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_dynamic_matrix_format_container<TransposeMatrix>::value>::
        type* = nullptr) {
  Impl::Variant::visit(
      [&](auto&& arg1, auto&& arg2) { Impl::transpose<ExecSpace>(arg1, arg2); },
      A.const_formats(), At.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP
