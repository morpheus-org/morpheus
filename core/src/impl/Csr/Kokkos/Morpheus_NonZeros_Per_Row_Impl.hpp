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

#ifndef MORPHEUS_CSR_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_CSR_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP

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
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using execution_space   = typename ExecSpace::execution_space;
  using size_type         = typename Matrix::size_type;
  using policy_index_type = Kokkos::IndexType<size_type>;
  using range_policy  = Kokkos::RangePolicy<policy_index_type, execution_space>;
  using offsets_array = typename Matrix::index_array_type::value_array_type;
  using vector_array  = typename Vector::value_array_type;

  using value_type = typename vector_array::value_type;
  using index_type = typename offsets_array::value_type;

  MORPHEUS_ASSERT(nnz_per_row.size() == A.nrows(),
                  "Destination vector must have equal size to the source "
                  "matrix number of rows");

  const offsets_array row_offsets = A.crow_offsets().const_view();
  vector_array nnz_per_row_view   = nnz_per_row.view();

  range_policy policy(0, A.nrows());

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_type i) {
        value_type non_zeros = init ? value_type(0) : nnz_per_row_view[i];
        for (index_type jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
          non_zeros++;
        }
        nnz_per_row_view[i] = non_zeros;
      });
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_KOKKOS_NON_ZEROS_PER_ROW_IMPL_HPP