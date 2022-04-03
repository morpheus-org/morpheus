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

#ifndef MORPHEUS_DIA_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DIA_KOKKOS_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <Morpheus_Exceptions.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, DiaTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using value_array_type =
      typename SparseMatrix::value_array_type::value_array_type;
  using index_array_type =
      typename SparseMatrix::index_array_type::value_array_type;
  using array_type = typename Vector::value_array_type;
  using index_type = typename SparseMatrix::index_type;
  using value_type = typename SparseMatrix::value_type;
  using range_policy =
      Kokkos::RangePolicy<Kokkos::IndexType<index_type>, execution_space>;

  value_array_type values           = A.values().view();
  index_array_type diagonal_offsets = A.diagonal_offsets().view();
  const array_type diagonal_view    = diagonal.const_view();

  index_type ndiag = A.values().ncols(), ncols = A.ncols();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const index_type row) {
        for (index_type n = 0; n < ndiag; n++) {
          const index_type col = row + diagonal_offsets[n];

          if ((col >= 0 && col < ncols) && (col == row)) {
            values(row, n) =
                (values(row, n) == value_type(0)) ? 0 : diagonal_view[col];
          }
        }
      });
}

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void get_diagonal(
    SparseMatrix& A, const Vector& diagonal, DiaTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename SparseMatrix, typename IndexType,
          typename ValueType>
void set_value(SparseMatrix& A, IndexType row, IndexType col, ValueType value,
               DiaTag,
               typename std::enable_if_t<
                   Morpheus::is_kokkos_space_v<ExecSpace> &&
                   Morpheus::has_access_v<typename ExecSpace::execution_space,
                                          SparseMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_value not implemented yet");
}

template <typename ExecSpace, typename SparseMatrix, typename IndexVector,
          typename ValueVector>
void set_values(
    SparseMatrix& A, typename IndexVector::value_type m, const IndexVector idxm,
    typename IndexVector::value_type n, const IndexVector idxn,
    const ValueVector values, DiaTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, IndexVector, ValueVector>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix& A, TransposeMatrix& At, DiaTag, DiaTag,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               TransposeMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
