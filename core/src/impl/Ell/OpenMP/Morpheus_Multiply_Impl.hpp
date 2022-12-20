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

#ifndef MORPHEUS_ELL_OPENMP_MULTIPLY_IMPL_HPP
#define MORPHEUS_ELL_OPENMP_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using size_type  = typename Matrix::size_type;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const size_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type invalid_index      = A.invalid_index();

  if (init) {
#pragma omp parallel for
    for (size_type n = 0; n < A.nrows(); n++) {
      y[n] = value_type(0);
    }
  }

#pragma omp parallel for
  for (size_type i = 0; i < A.nrows(); i++) {
    for (size_type n = 0; n < num_entries_per_row; n++) {
      if (A.ccolumn_indices(i, n) != invalid_index) {
        y[i] += A.cvalues(i, n) * x[A.ccolumn_indices(i, n)];
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ELL_OPENMP_MULTIPLY_IMPL_HPP