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

#ifndef MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecutionSpace, typename LinearOperator,
          typename MatrixOrVector1, typename MatrixOrVector2>
struct DiaSpmv_Kokkos {
  using member_type = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  using value_array_type =
      typename LinearOperator::value_array_type::value_array_type;
  using index_array_type =
      typename LinearOperator::index_array_type::value_array_type;
  using array_type1 = typename MatrixOrVector1::value_array_type;
  using array_type2 = typename MatrixOrVector2::value_array_type;
  using index_type  = typename LinearOperator::index_type;
  using value_type  = typename LinearOperator::value_type;

  value_array_type values;
  array_type1 x;
  array_type2 y;
  index_array_type diagonal_offsets;
  index_type ndiag, ncols;

  DiaSpmv_Kokkos(LinearOperator A, MatrixOrVector1 _x, MatrixOrVector2 _y)
      : values(A.values.const_view()),
        diagonal_offsets(A.diagonal_offsets.const_view()),
        x(_x.const_view()),
        y(_y.view()) {
    ndiag = A.values.ncols();
    ncols = A.ncols();
  }

  KOKKOS_INLINE_FUNCTION void operator()(const member_type& team_member) const {
    const index_type row = team_member.league_rank();

    value_type sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, ndiag),
        [=](const int& n, value_type& lsum) {
          const index_type col = row + diagonal_offsets[n];

          if (col >= 0 && col < ncols) {
            lsum += values(row, n) * x[col];
          }
        },
        sum);
    y[row] = sum;
  }
};

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    DiaTag, DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  using execution_space = typename ExecSpace::execution_space;

  const Kokkos::TeamPolicy<execution_space> policy(A.nrows(), Kokkos::AUTO,
                                                   Kokkos::AUTO);

  Kokkos::parallel_for(
      policy, DiaSpmv_Kokkos<execution_space, LinearOperator, MatrixOrVector1,
                             MatrixOrVector2>(A, x, y));
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP