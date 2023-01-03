/**
 * Morpheus_MatrixAnalytics_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_MATRIXANALYTICS_IMPL_HPP
#define MORPHEUS_DYNAMIC_MATRIXANALYTICS_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_MatrixAnalytics_Impl.hpp>
#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void count_nnz_per_row(
    const Matrix& A, Vector& nnz_per_row, const bool init,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<Matrix>::value &&
        Morpheus::is_dense_vector_format_container<Vector>::value>::type* =
        nullptr) {
  std::visit(
      [&](auto&& arg) {
        Impl::count_nnz_per_row<ExecSpace>(arg, nnz_per_row, init);
      },
      A.const_formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_MATRIXANALYTICS_IMPL_HPP