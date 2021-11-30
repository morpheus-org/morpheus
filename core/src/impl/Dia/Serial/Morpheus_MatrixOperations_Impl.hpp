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

#ifndef MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SparseMatrix, typename Vector>
void update_diagonal(
    SparseMatrix& A, const Vector& diagonal, DiaTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               SparseMatrix, Vector>>* = nullptr) {
  using IndexType = typename SparseMatrix::index_type;
  using ValueType = typename SparseMatrix::value_type;

  const IndexType ndiag = A.values().ncols();

  for (IndexType row = 0; row < A.nrows(); row++) {
    for (IndexType n = 0; n < ndiag; n++) {
      const IndexType col = row + A.diagonal_offsets(n);

      if ((col >= 0 && col < A.ncols()) && (col == row)) {
        A.values(row, n) =
            (A.values(row, n) == ValueType(0)) ? 0 : diagonal[col];
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_SERIAL_MATRIXOPERATIONS_IMPL_HPP
