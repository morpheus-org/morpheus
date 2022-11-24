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

#ifndef MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Scan.hpp>

#include <impl/Morpheus_OpenMPUtils.hpp>

#include <limits>

namespace Morpheus {
namespace Impl {

template <typename T>
int is_row_stop(T container, typename T::index_type start_idx,
                typename T::index_type end_idx) {
  return container[start_idx] != container[end_idx];
}

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  if (init) {
#pragma omp parallel for
    for (index_type n = 0; n < A.nrows(); n++) {
      y[n] = value_type(0);
    }
  }

#pragma omp parallel
  {
    const size_t num_threads     = omp_get_num_threads();
    const size_t work_per_thread = (A.nnnz() + num_threads - 1) / num_threads;
    const size_t thread_id       = omp_get_thread_num();
    const size_t begin           = work_per_thread * thread_id;
    const size_t end = std::min(begin + work_per_thread, (size_t)A.nnnz());
    const auto sentinel_row = A.nrows() + 1;

    if (begin < end) {
      const auto first = begin > 0 ? A.crow_indices(begin - 1) : sentinel_row;
      const auto last  = end < A.nnnz() ? A.crow_indices(end) : sentinel_row;
      auto n           = begin;

      // handle non-overlapping rows
      for (; n < end && A.crow_indices(n) != last; n++) {
        y[A.crow_indices(n)] += A.cvalues(n) * x[A.ccolumn_indices(n)];
      }

      // handle row overlap with previous thread
      if (first != sentinel_row) {
        value_type partial_sum = 0;
        for (; n < end && A.crow_indices(n) == first; n++) {
          partial_sum += A.cvalues(n) * x[A.ccolumn_indices(n)];
        }
#pragma omp atomic
        y[first] += partial_sum;
      }

      // handle row overlap with following thread
      if (last != sentinel_row) {
        value_type partial_sum = 0;
        for (; n < end; n++) {
          partial_sum += A.cvalues(n) * x[A.ccolumn_indices(n)];
        }
#pragma omp atomic
        y[last] += partial_sum;
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP