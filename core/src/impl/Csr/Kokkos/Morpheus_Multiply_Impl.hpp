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

#ifndef MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>
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
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector1, Vector2>>* = nullptr) {
  using execution_space   = typename ExecSpace::execution_space;
  using policy_index_type = Kokkos::IndexType<typename Matrix::index_type>;
  using range_policy = Kokkos::RangePolicy<policy_index_type, execution_space>;
  using value_array  = typename Matrix::value_array_type::value_array_type;
  using index_array  = typename Matrix::index_array_type::value_array_type;
  using value_type   = typename value_array::value_type;
  using index_type   = typename index_array::value_type;

  const value_array values = A.cvalues().const_view(), x_view = x.const_view();
  const index_array column_indices = A.ccolumn_indices().const_view(),
                    row_offsets    = A.crow_offsets().const_view();
  value_array y_view               = y.view();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const index_type i) {
        value_type sum = init ? value_type(0) : y_view[i];
        for (index_type jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
          sum += values[jj] * x_view[column_indices[jj]];
        }
        y_view[i] = sum;
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP