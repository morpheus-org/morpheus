/**
 * Morpheus_MatrixOperations_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
// fwd decl
template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal);

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void get_diagonal(const SparseMatrix& A, Vector& diagonal);

template <typename ExecSpace, typename SparseMatrix, typename IndexType,
          typename ValueType>
void set_value(SparseMatrix& A, IndexType row, IndexType col, ValueType value);

template <typename ExecSpace, typename SparseMatrix, typename IndexVector,
          typename ValueVector>
void set_values(SparseMatrix& A, typename IndexVector::value_type m,
                const IndexVector idxm, typename IndexVector::value_type n,
                const IndexVector idxn, ValueVector values);

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(const Matrix& A, TransposeMatrix& At);

namespace Impl {
template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal,
                            Morpheus::DynamicTag, Morpheus::DenseVectorTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg) { Morpheus::update_diagonal<ExecSpace>(arg, diagonal); },
      A.formats());
}

template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void get_diagonal(const SparseMatrix& A, Vector& diagonal,
                         Morpheus::DynamicTag, Morpheus::DenseVectorTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg) { Morpheus::get_diagonal<ExecSpace>(arg, diagonal); },
      A.const_formats());
}

template <typename ExecSpace, typename SparseMatrix, typename IndexType,
          typename ValueType>
inline void set_value(SparseMatrix& A, IndexType row, IndexType col,
                      ValueType value, Morpheus::DynamicTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg) { Morpheus::set_value<ExecSpace>(arg, row, col, value); },
      A.formats());
}

template <typename ExecSpace, typename SparseMatrix, typename IndexVector,
          typename ValueVector>
inline void set_values(SparseMatrix& A, typename IndexVector::value_type m,
                       const IndexVector idxm,
                       typename IndexVector::value_type n,
                       const IndexVector idxn, ValueVector values,
                       Morpheus::DynamicTag, Morpheus::DenseVectorTag,
                       Morpheus::DenseVectorTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg) {
        Morpheus::set_value<ExecSpace>(arg, m, idxm, n, idxn, values);
      },
      A.formats());
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
inline void transpose(const Matrix& A, TransposeMatrix& At,
                      Morpheus::DynamicTag, Impl::SparseMatTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg) { Morpheus::transpose<ExecSpace>(arg, At); },
      A.const_formats());
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
inline void transpose(const Matrix& A, TransposeMatrix& At,
                      Morpheus::DynamicTag, Morpheus::DynamicTag) {
  Morpheus::Impl::Variant::visit(
      [&](auto&& arg1, auto&& arg2) {
        Morpheus::transpose<ExecSpace>(arg1, arg2);
      },
      A.const_formats(), A.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP
