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

template <typename ValueArray, typename IndexArray>
struct csr_spmv_inner_functor {
  ValueArray values, x_view, y_view;
  IndexArray column_indices, row_offsets;
  using V = typename ValueArray::value_type;
  using I = typename IndexArray::value_type;

  csr_spmv_inner_functor(IndexArray _row_offsets, IndexArray _column_indices,
                         ValueArray _values, ValueArray _x_view,
                         ValueArray _y_view)
      : row_offsets(_row_offsets),
        column_indices(_column_indices),
        values(_values),
        x_view(_x_view),
        y_view(_y_view) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const I& i) const {
    V sum = y_view[i];
    for (I jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
      sum += values[jj] * x_view[column_indices[jj]];
    }
    y_view[i] = sum;
  }
};
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

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy,
      csr_spmv_inner_functor(A.row_offsets.view(), A.column_indices.view(),
                             A.values.view(), x.view(), y.view()));
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_MULTIPLY_IMPL_HPP