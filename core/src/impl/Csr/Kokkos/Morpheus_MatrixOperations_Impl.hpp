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

#ifndef MORPHEUS_CSR_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_CSR_KOKKOS_MATRIXOPERATIONS_IMPL_HPP

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
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using index_type      = typename Matrix::index_type;
  using value_array     = typename Matrix::value_array_type::value_array_type;
  using index_array     = typename Matrix::index_array_type::value_array_type;

  using range_policy =
      Kokkos::RangePolicy<Kokkos::IndexType<index_type>, execution_space>;

  value_array values              = A.values().view();
  index_array column_indices      = A.column_indices().view();
  index_array row_offsets         = A.row_offsets().view();
  const value_array diagonal_view = diagonal.const_view();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const index_type i) {
        for (index_type jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
          if (column_indices[jj] == i) {
            values[jj] = diagonal_view[i];
            break;
          }
        }
      });
}

template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename IndexType,
          typename ValueType>
void set_value(
    Matrix& A, IndexType row, IndexType col, ValueType value,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
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
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<IndexVector> &&
        Morpheus::is_dense_vector_format_container_v<ValueVector> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               IndexVector, ValueVector>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix& A, TransposeMatrix& At,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_csr_matrix_format_container_v<TransposeMatrix> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               TransposeMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
