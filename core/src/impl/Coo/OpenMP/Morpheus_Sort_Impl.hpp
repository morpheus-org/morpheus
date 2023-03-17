/**
 * Morpheus_Sort_Impl.hpp
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

#ifndef MORPHEUS_COO_OPENMP_SORT_IMPL_HPP
#define MORPHEUS_COO_OPENMP_SORT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix>
void sort_by_row_and_column(
    Matrix&, typename Matrix::index_type = 0, typename Matrix::index_type = 0,
    typename Matrix::index_type = 0, typename Matrix::index_type = 0,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  throw Morpheus::NotImplementedException(
      "Impl.Coo.OpenMP.sort_by_row_and_column()");
}

template <typename ExecSpace, typename Matrix>
bool is_sorted(Matrix&,
               typename std::enable_if_t<
                   Morpheus::is_coo_matrix_format_container_v<Matrix> &&
                   Morpheus::has_custom_backend_v<ExecSpace> &&
                   Morpheus::has_openmp_execution_space_v<ExecSpace> &&
                   Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("Impl.Coo.OpenMP.is_sorted()");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_SORT_IMPL_HPP