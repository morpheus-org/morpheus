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

#ifndef MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CsrTag, DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;
  using index_type   = Kokkos::IndexType<typename LinearOperator::index_type>;
  using range_policy = Kokkos::RangePolicy<index_type, execution_space>;
  using ValueArray =
      typename LinearOperator::value_array_type::value_array_type;
  using IndexArray =
      typename LinearOperator::index_array_type::value_array_type;
  using V = typename ValueArray::value_type;
  using I = typename IndexArray::value_type;

  const ValueArray values = A.values.const_view(), x_view = x.const_view();
  const IndexArray column_indices = A.column_indices.const_view(),
                   row_offsets    = A.row_offsets.const_view();
  ValueArray y_view               = y.view();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const I i) {
        V sum = 0;
        for (I jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
          sum += values[jj] * x_view[column_indices[jj]];
        }
        y_view[i] = sum;
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP