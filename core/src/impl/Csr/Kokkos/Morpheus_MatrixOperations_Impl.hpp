/**
 * Morpheus_MatrixOperations_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
inline void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, CsrTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using index_type      = typename SparseMatrix::index_type;
  using ValueArray = typename SparseMatrix::value_array_type::value_array_type;
  using IndexArray = typename SparseMatrix::index_array_type::value_array_type;

  using range_policy =
      Kokkos::RangePolicy<Kokkos::IndexType<index_type>, execution_space>;

  ValueArray values              = A.values.view();
  IndexArray column_indices      = A.column_indices.view();
  IndexArray row_offsets         = A.row_offsets.view();
  const ValueArray diagonal_view = diagonal.const_view();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const index_type i) {
        for (index_type jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
          if (column_indices[jj] == i) values[jj] = diagonal_view[i];
        }
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_MATRIXOPERATIONS_IMPL_HPP
