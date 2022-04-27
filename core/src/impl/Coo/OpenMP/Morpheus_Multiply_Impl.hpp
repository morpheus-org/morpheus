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

template <typename ExecSpace, typename Matrix, typename Vector1,
          typename Vector2>
inline void multiply(
    const Matrix& A, const Vector1& x, Vector2& y, const bool init, CooTag,
    DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector1, Vector2>>* = nullptr) {
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  const index_type max_threads =
      A.nnnz() < threads<index_type>() ? A.nnnz() : threads<index_type>();

  if (init) {
#pragma omp parallel for
    for (index_type n = 0; n < A.nrows(); n++) {
      y[n] = value_type(0);
    }
  }

#pragma omp parallel num_threads(max_threads)
  {
    const index_type nthreads = omp_get_num_threads();
    const index_type tid      = omp_get_thread_num();

    const index_type thread_start = _split_work(A.nnnz(), nthreads, tid);
    const index_type thread_stop  = _split_work(A.nnnz(), nthreads, tid + 1);

    index_type first_row_start = thread_stop, last_row_stop = thread_stop;

    for (index_type n = thread_start; n < thread_stop - 1; n++) {
      if (A.crow_indices(n) != A.crow_indices(n + 1)) {
        first_row_start = n;
        break;
      }
    }

    for (index_type n = thread_stop; n < thread_start + 1; n--) {
      if (A.crow_indices(n) != A.crow_indices(n - 1)) {
        last_row_stop = n;
        break;
      }
    }

    value_type temp = value_type(0);
    for (index_type n = thread_start; n < first_row_start; n++) {
      temp += A.cvalues(n) * x(A.ccolumn_indices(n));
    }

#pragma omp atomic
    y[A.crow_indices(thread_start)] += temp;

    for (index_type n = first_row_start; n < last_row_stop; n++) {
      y[A.crow_indices(n)] += A.cvalues(n) * x(A.ccolumn_indices(n));
    }

    temp = value_type(0);
    for (index_type n = last_row_stop; n < thread_stop; n++) {
      temp += A.cvalues(n) * x(A.ccolumn_indices(n));
    }

#pragma omp atomic
    y[A.crow_indices(thread_stop)] += temp;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP