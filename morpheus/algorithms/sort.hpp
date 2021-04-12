/**
 * sort.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#ifndef MORPHEUS_ALGORITHMS_SORT_HPP
#define MORPHEUS_ALGORITHMS_SORT_HPP

#include <morpheus/algorithms/impl/sort_impl.hpp>

namespace Morpheus {

template <typename ExecSpace, typename Matrix>
void sort_by_row_and_column(const ExecSpace& space, Matrix& mat,
                            typename Matrix::index_type min_row = 0,
                            typename Matrix::index_type max_row = 0,
                            typename Matrix::index_type min_col = 0,
                            typename Matrix::index_type max_col = 0) {
  Impl::sort_by_row_and_column(space, mat, typename Matrix::tag{}, min_row,
                               max_row, min_col, max_col);
}

template <typename ExecSpace, typename Matrix>
bool is_sorted(const ExecSpace& space, Matrix& mat) {
  return Impl::is_sorted(space, mat, typename Matrix::tag{});
}

}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_SORT_HPP