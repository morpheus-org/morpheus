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

#ifndef MORPHEUS_ELL_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_ELL_KOKKOS_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void update_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using size_type       = typename Matrix::size_type;
  using index_type      = typename Matrix::index_type;
  using value_type      = typename Matrix::value_type;
  using value_array     = typename Matrix::value_array_type::value_array_type;
  using index_array     = typename Matrix::index_array_type::value_array_type;
  using diagonal_array  = typename Vector::value_array_type;

  using range_policy =
      Kokkos::RangePolicy<Kokkos::IndexType<size_type>, execution_space>;

  value_array values                  = A.values().view();
  index_array column_indices          = A.column_indices().view();
  const diagonal_array diagonal_view  = diagonal.const_view();
  const size_type num_entries_per_row = A.entries_per_row();
  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_type i) {
        for (size_type n = 0; n < num_entries_per_row; n++) {
          if (column_indices(i, n) == (index_type)i) {
            values(i, n) =
                (values(i, n) == value_type(0)) ? 0 : diagonal_view[i];
            break;
          }
        }
      });
}

template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(
    Matrix&, const Vector&,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename SizeType,
          typename ValueType>
void set_value(Matrix&, SizeType, SizeType, ValueType,
               typename std::enable_if_t<
                   Morpheus::is_ell_matrix_format_container_v<Matrix> &&
                   Morpheus::has_generic_backend_v<ExecSpace> &&
                   Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_value not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
void set_values(
    Matrix&, typename IndexVector::value_type, const IndexVector,
    typename IndexVector::value_type, const IndexVector, const ValueVector,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<IndexVector> &&
        Morpheus::is_dense_vector_format_container_v<ValueVector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, IndexVector, ValueVector>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix&, TransposeMatrix&,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_ell_matrix_format_container_v<TransposeMatrix> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, TransposeMatrix>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ELL_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
