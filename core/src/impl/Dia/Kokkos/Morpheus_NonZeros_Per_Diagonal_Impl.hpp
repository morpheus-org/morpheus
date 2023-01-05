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

#ifndef MORPHEUS_DIA_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP
#define MORPHEUS_DIA_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP

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
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space   = typename ExecSpace::execution_space;
  using size_type         = typename Matrix::size_type;
  using policy_index_type = Kokkos::IndexType<size_type>;
  using range_policy = Kokkos::RangePolicy<policy_index_type, execution_space>;
  using value_array_type = typename Matrix::value_array_type::value_array_type;
  using index_array_type = typename Matrix::index_array_type::value_array_type;

  MORPHEUS_ASSERT(nnz_per_diagonal.size() == A.nrows() + A.ncols() - 1,
                  "Destination vector must have equal size to the source "
                  "matrix number of diagonals (i.e NROWS + NCOLS - 1)");

  const value_array_type values           = A.cvalues().const_view();
  const index_array_type diagonal_offsets = A.cdiagonal_offsets().const_view();
  typename Vector::value_array_type out_view = nnz_per_diagonal.view();
  size_type ndiag                            = A.cdiagonal_offsets().size();

  if (init) {
    nnz_per_diagonal.assign(nnz_per_diagonal.size(), 0);
  }

  range_policy policy(0, ndiag);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_type i) {
        for (size_type j = 0; j < values.extent(0); j++) {
          auto diag_idx = diagonal_offsets[i] + A.nrows() - 1;
          if (values(j, i) != 0) {
            out_view[diag_idx] += 1;
          }
        }
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_KOKKOS_NON_ZEROS_PER_DIAGONAL_IMPL_HPP