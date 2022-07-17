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

#ifndef MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_GenericSpace.hpp>
#include <Morpheus_Exceptions.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void update_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const index_type ndiag = A.values().ncols();

  for (index_type row = 0; row < A.nrows(); row++) {
    for (index_type n = 0; n < ndiag; n++) {
      const index_type col = row + A.diagonal_offsets(n);

      if ((col >= 0 && col < A.ncols()) && (col == row)) {
        A.values(row, n) =
            (A.values(row, n) == value_type(0)) ? 0 : diagonal[col];
      }
    }
  }
}

template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename IndexType,
          typename ValueType>
void set_value(
    Matrix& A, IndexType row, IndexType col, ValueType value,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("set_value not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
void set_values(
    Matrix& A, typename IndexVector::value_type m, const IndexVector idxm,
    typename IndexVector::value_type n, const IndexVector idxn,
    const ValueVector values,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<IndexVector> &&
        Morpheus::is_dense_vector_format_container_v<ValueVector> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               IndexVector, ValueVector>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix& A, TransposeMatrix& At,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dia_matrix_format_container_v<TransposeMatrix> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               TransposeMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP
