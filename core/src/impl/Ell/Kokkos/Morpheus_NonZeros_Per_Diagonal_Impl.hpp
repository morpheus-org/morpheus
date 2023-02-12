/**
 * Morpheus_NonZeros_Per_Diagonal_Impl.hpp
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

#ifndef MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP
#define MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void count_nnz_per_diagonal(
    const Matrix& A, Vector& nnz_per_diagonal, const bool init,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space   = typename ExecSpace::execution_space;
  using size_type         = typename Matrix::size_type;
  using index_type        = typename Matrix::index_type;
  using policy_index_type = Kokkos::IndexType<size_type>;
  using range_policy = Kokkos::RangePolicy<policy_index_type, execution_space>;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;
  using value_array_type = typename Matrix::value_array_type::value_array_type;
  using index_array_type = typename Matrix::index_array_type::value_array_type;
  using IndexVector =
      Morpheus::DenseVector<index_type, size_type, typename Matrix::backend>;

  MORPHEUS_ASSERT(nnz_per_diagonal.size() == A.nrows() + A.ncols() - 1,
                  "Destination vector must have equal size to the source "
                  "matrix number of diagonals (i.e NROWS + NCOLS - 1)");

  const value_array_type values         = A.cvalues().const_view();
  const index_array_type column_indices = A.ccolumn_indices().const_view();

  const size_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type invalid_index      = A.invalid_index();

  if (init) {
    nnz_per_diagonal.assign(nnz_per_diagonal.size(), 0);
  }

  const Kokkos::TeamPolicy<execution_space> policy(A.nrows(), Kokkos::AUTO,
                                                   Kokkos::AUTO);
  Vector row_offsets(A.nrows() + 1, 0), mirror_offsets(A.nrows() + 1, 0);
  typename Vector::value_array_type offsets_view        = row_offsets.view();
  typename Vector::value_array_type mirror_offsets_view = mirror_offsets.view();
  // count per row the number of nnz
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const member_type& team_member) {
        const index_type row = team_member.league_rank();

        size_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, num_entries_per_row),
            [=](const size_type& n, size_type& lsum) {
              const index_type col = column_indices(row, n);
              if (col != invalid_index && values(row, n) != 0) {
                lsum += 1;
              }
            },
            sum);

        team_member.team_barrier();
        if (team_member.team_rank() == 0) {
          offsets_view[row] = sum;
        };
      });

  // generate cumulative nnnz per row
  size_type result;
  Kokkos::parallel_scan(
      "accumulate nnz per row", A.nrows() + 1,
      KOKKOS_LAMBDA(size_type i, size_type & partial_sum, bool is_final) {
        if (is_final) mirror_offsets_view[i] = partial_sum;
        partial_sum += offsets_view[i];
      },
      result);

  // parallelize over rows to find the nnnz per diagonal
  size_type nrows = A.nrows();
  typename Vector::value_array_type nnz_per_diagonal_view =
      nnz_per_diagonal.view();
  range_policy range(0, A.nrows());

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(const size_type i) {
        for (index_type jj = mirror_offsets_view[i];
             jj < mirror_offsets_view[i + 1]; jj++) {
          const index_type col = column_indices(i, jj - mirror_offsets_view[i]);
          if (col != invalid_index &&
              values(i, jj - mirror_offsets_view[i]) != 0) {
            // Diagonal index is offseted by the number of rows
            size_type idx = col - i + nrows - 1;
            Kokkos::atomic_increment(&nnz_per_diagonal_view(idx));
          }
        }
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ELL_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP