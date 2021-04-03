/**
 * multiply_serial.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_SERIAL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_SERIAL_HPP

#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/vector.hpp>

namespace Morpheus {
namespace Impl {

template <typename Matrix, typename Vector>
void multiply(const Matrix& A, const Vector& x, Vector& y, Morpheus::CsrTag,
              typename std::enable_if<
                  std::is_same<typename Matrix::execution_space,
                               Kokkos::Serial::execution_space>::value,
                  Kokkos::Serial::execution_space>::type* = nullptr) {
  // Check all containers have access to the same execution space
  static_assert(std::is_same_v<typename Matrix::execution_space,
                               typename Vector::execution_space>);

  using I = typename Matrix::index_type;
  using T = typename Matrix::value_type;

  for (I i = 0; i < A.nrows(); i++) {
    T sum = y[i];
    for (I jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++) {
      sum += A.values[jj] * x[A.column_indices[jj]];
    }
    y[i] = sum;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_SERIAL_HPP