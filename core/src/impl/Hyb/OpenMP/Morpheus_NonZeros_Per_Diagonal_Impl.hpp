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

#ifndef MORPHEUS_HYB_OPENMP_NON_ZEROS_PER_DIAGONAL_IMPL_HPP
#define MORPHEUS_HYB_OPENMP_NON_ZEROS_PER_DIAGONAL_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/Ell/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void count_nnz_per_diagonal(
    const Matrix& A, Vector& nnz_per_diagonal, const bool init,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  MORPHEUS_ASSERT(nnz_per_diagonal.size() == A.nrows() + A.ncols() - 1,
                  "Destination vector must have equal size to the source "
                  "matrix number of diagonals (i.e NROWS + NCOLS - 1)");

  Impl::count_nnz_per_diagonal<ExecSpace>(A.cell(), nnz_per_diagonal, init);
  Impl::count_nnz_per_diagonal<ExecSpace>(A.ccoo(), nnz_per_diagonal, false);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_HYB_OPENMP_NON_ZEROS_PER_DIAGONAL_IMPL_HPP