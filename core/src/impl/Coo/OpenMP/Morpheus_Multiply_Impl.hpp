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

#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>
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
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_openmp_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using size_type  = typename Matrix::size_type;
  using value_type = typename Vector::value_type;
  using index_type = typename Matrix::index_type;

  if (init) {
    y.assign(y.size(), 0);
  }

  const size_type nrows        = A.nrows();
  const size_type nnnz         = A.nnnz();
  const size_type sentinel_row = nrows + 1;

  index_type* rind = A.crow_indices().data();
  index_type* cind = A.ccolumn_indices().data();
  value_type* Aval = A.cvalues().data();
  value_type* xval = x.data();
  value_type* yval = y.data();

#pragma omp parallel
  {
    const size_type num_threads = omp_get_num_threads();
    const size_type work_per_thread =
        Impl::ceil_div<size_type>(nnnz, num_threads);
    const size_type thread_id = omp_get_thread_num();
    const size_type begin     = work_per_thread * thread_id;
    const size_type end       = std::min(begin + work_per_thread, nnnz);
    size_type n               = begin;

    if (begin < end) {
      const index_type first = begin > 0 ? rind[begin - 1] : sentinel_row;
      const index_type last  = end < nnnz ? rind[end] : sentinel_row;

      // handle row overlap with previous thread
      if (first != (index_type)sentinel_row) {
        value_type partial_sum = value_type(0);
        for (; n < end && rind[n] == first; n++) {
          partial_sum += Aval[n] * xval[cind[n]];
        }
        Impl::atomic_add(&yval[first], partial_sum);
      }

      // handle non-overlapping rows
      for (; n < end && A.crow_indices(n) != last; n++) {
        yval[rind[n]] += Aval[n] * xval[cind[n]];
      }

      // handle row overlap with following thread
      if (last != (index_type)sentinel_row) {
        value_type partial_sum = value_type(0);
        for (; n < end; n++) {
          partial_sum += Aval[n] * xval[cind[n]];
        }
        Impl::atomic_add(&yval[last], partial_sum);
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP