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

#ifndef MORPHEUS_HDC_SERIAL_NON_ZEROS_PER_ROW_IMPL_HPP
#define MORPHEUS_HDC_SERIAL_NON_ZEROS_PER_ROW_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_Utils.hpp>
#include <impl/Dia/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if_t<
        Morpheus::is_hdc_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  MORPHEUS_ASSERT(nnz_per_row.size() == A.nrows(),
                  "Destination vector must have equal size to the source "
                  "matrix number of rows");

  Impl::count_nnz_per_row<ExecSpace>(A.cdia(), nnz_per_row, init);
  Impl::count_nnz_per_row<ExecSpace>(A.ccsr(), nnz_per_row, false);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_HDC_SERIAL_NON_ZEROS_PER_ROW_IMPL_HPP