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

#ifndef MORPHEUS_COO_OPENMP_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_COO_OPENMP_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <Morpheus_Exceptions.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, CooTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using index_type = typename SparseMatrix::index_type;

// Note: Assumes only a single entry exists for every diagonal element.
// i.e a single threads will update a particular non-zero on the diagonal
#pragma omp parallel for
  for (index_type n = 0; n < A.nnnz(); n++) {
    if (A.row_indices(n) == A.column_indices(n)) {
      A.values(n) = diagonal[A.column_indices(n)];
    }
  }
}

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void get_diagonal(
    SparseMatrix& A, const Vector& diagonal, CooTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename SparseMatrix, typename IndexType,
          typename ValueType>
void set_value(SparseMatrix& A, IndexType row, IndexType col, ValueType value,
               CooTag,
               typename std::enable_if_t<
                   !Morpheus::is_kokkos_space_v<ExecSpace> &&
                   Morpheus::is_openmp_execution_space_v<ExecSpace> &&
                   Morpheus::has_access_v<typename ExecSpace::execution_space,
                                          SparseMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_value not implemented yet");
}

template <typename ExecSpace, typename SparseMatrix, typename IndexVector,
          typename ValueVector>
void set_values(
    SparseMatrix& A, typename IndexVector::value_type m, const IndexVector idxm,
    typename IndexVector::value_type n, const IndexVector idxn,
    const ValueVector values, CooTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, IndexVector, ValueVector>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix& A, TransposeMatrix& At, CooTag, CooTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               TransposeMatrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MATRIXOPERATIONS_IMPL_HPP
