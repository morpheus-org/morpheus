/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_CSR_SERIAL_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_SERIAL_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector1,
          typename Vector2>
inline void multiply(
    const Matrix& A, const Vector1& x, Vector2& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        !Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::is_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector1, Vector2>>* = nullptr) {
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  for (index_type i = 0; i < A.nrows(); i++) {
    value_type sum = init ? value_type(0) : y[i];
    for (index_type jj = A.crow_offsets(i); jj < A.crow_offsets(i + 1); jj++) {
      sum += A.cvalues(jj) * x[A.ccolumn_indices(jj)];
    }
    y[i] = sum;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_SERIAL_MULTIPLY_IMPL_HPP