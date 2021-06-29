/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_SERIAL_MULTIPLY_IMPL_HPP
#define MORPHEUS_SERIAL_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <impl/Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(
    const ExecSpace& space, const LinearOperator& A, const MatrixOrVector1& x,
    MatrixOrVector2& y, CooTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using I = typename LinearOperator::index_type;

  for (I n = 0; n < A.nnnz(); n++) {
    y[A.row_indices[n]] += A.values[n] * x[A.column_indices[n]];
  }
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(
    const ExecSpace& space, const LinearOperator& A, const MatrixOrVector1& x,
    MatrixOrVector2& y, CsrTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using I = typename LinearOperator::index_type;
  using T = typename LinearOperator::value_type;

  for (I i = 0; i < A.nrows(); i++) {
    T sum = y[i];
    for (I jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++) {
      sum += A.values[jj] * x[A.column_indices[jj]];
    }
    y[i] = sum;
  }
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(
    const ExecSpace& space, const LinearOperator& A, const MatrixOrVector1& x,
    MatrixOrVector2& y, DiaTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_Serial_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using I = typename LinearOperator::index_type;

  for (I i = 0; i < (int)A.diagonal_offsets.size(); i++) {
    const I k       = A.diagonal_offsets[i];  // diagonal offset
    const I i_start = std::max(0, -k);
    const I j_start = std::max(0, k);
    const I N       = std::min(A.nrows() - i_start, A.ncols() - j_start);

    for (I n = 0; n < N; n++) {
      y[i_start + n] += A.values(i, j_start + n) * x[j_start + n];
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_SERIAL_MULTIPLY_IMPL_HPP