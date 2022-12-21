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

#ifndef MORPHEUS_ELL_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_ELL_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Ell/Serial/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space  = typename ExecSpace::execution_space;
  using value_array_type = typename Matrix::value_array_type::value_array_type;
  using index_array_type = typename Matrix::index_array_type::value_array_type;
  using size_type        = typename Matrix::size_type;
  using index_type       = typename Matrix::index_type;
  using value_type       = typename Matrix::value_type;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  const value_array_type values         = A.cvalues().const_view();
  const index_array_type column_indices = A.ccolumn_indices().const_view();
  const typename Vector::value_array_type x_view = x.const_view();
  typename Vector::value_array_type y_view       = y.view();

  const size_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type invalid_index      = A.invalid_index();

  if (init) {
    y.assign(y.size(), 0);
  }

  const Kokkos::TeamPolicy<execution_space> policy(A.nrows(), Kokkos::AUTO,
                                                   Kokkos::AUTO);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const member_type& team_member) {
        const index_type row = team_member.league_rank();

        value_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, num_entries_per_row),
            [=](const size_type& n, value_type& lsum) {
              const index_type col = column_indices(row, n);
              if (col != invalid_index) {
                lsum += values(row, n) * x_view[col];
              }
            },
            sum);

        team_member.team_barrier();
        if (team_member.team_rank() == 0) {
          y_view[row] = init ? sum : sum + y_view[row];
        };
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ELL_KOKKOS_MULTIPLY_IMPL_HPP