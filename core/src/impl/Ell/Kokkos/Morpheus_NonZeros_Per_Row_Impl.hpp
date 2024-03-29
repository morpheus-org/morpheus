/**
 * Morpheus_NonZeros_Per_Row_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space  = typename ExecSpace::execution_space;
  using index_array_type = typename Matrix::index_array_type::value_array_type;
  using size_type        = typename Matrix::size_type;
  using index_type       = typename Matrix::index_type;
  using value_type       = typename Vector::value_type;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  MORPHEUS_ASSERT(nnz_per_row.size() == A.nrows(),
                  "Destination vector must have equal size to the source "
                  "matrix number of rows");

  const index_array_type column_indices = A.ccolumn_indices().const_view();
  typename Vector::value_array_type nnz_per_row_view = nnz_per_row.view();

  const size_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type invalid_index      = A.invalid_index();

  if (init) {
    nnz_per_row.assign(nnz_per_row.size(), 0);
  }

  const Kokkos::TeamPolicy<execution_space> policy(A.nrows(), Kokkos::AUTO,
                                                   Kokkos::AUTO);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const member_type& team_member) {
        const index_type row = team_member.league_rank();

        value_type non_zeros = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, num_entries_per_row),
            [=](const size_type& n, value_type& lnon_zeros) {
              const index_type col = column_indices(row, n);
              if (col != invalid_index) {
                lnon_zeros++;
              }
            },
            non_zeros);

        team_member.team_barrier();
        if (team_member.team_rank() == 0) {
          nnz_per_row_view[row] =
              init ? non_zeros : non_zeros + nnz_per_row_view[row];
        };
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP