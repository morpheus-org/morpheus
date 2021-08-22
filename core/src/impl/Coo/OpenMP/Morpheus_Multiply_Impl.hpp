/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>
#include <Morpheus_Scan.hpp>

#include <limits>

namespace Morpheus {
namespace Impl {

template <typename T>
T _split_work(const T load, const T workers, const T worker_id) {
  const T unifload = load / workers;  // uniform distribution
  const T rem      = load - unifload * workers;
  T bound;

  //  round-robin assignment of the remaining work
  if (worker_id <= rem) {
    bound = (unifload + 1) * worker_id;
  } else {
    bound = (unifload + 1) * rem + unifload * (worker_id - rem);
  }

  return bound;
}

template <typename T>
T threads() {
  T t = 1;
#pragma omp parallel
  { t = omp_get_num_threads(); }

  return t;
}

template <typename T>
int is_row_stop(T container, typename T::index_type start_idx,
                typename T::index_type cur_index) {
  return container[start_idx] != container[cur_index];
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CooTag, DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  using ValueType = typename LinearOperator::value_type;
  using IndexType = typename LinearOperator::index_type;
  using KeyType   = typename LinearOperator::index_type;
  using vector    = MatrixOrVector1;

  if (A.nnnz() < threads<IndexType>()) {
    omp_set_num_threads(A.nnnz());
  }
  const IndexType nthreads = threads<IndexType>();
  const ValueType max_val  = std::numeric_limits<ValueType>::max();
  vector out("out", A.nnnz(), 0);
  // Initialize to special value
  vector grp_sum(nthreads, max_val);

#pragma omp parallel
  {
    const IndexType tid = omp_get_thread_num();

    // evenly divide workload based on nnz elements
    // each workgroup is assigned to one thread
    const IndexType chunk = A.nnnz() / nthreads;
    const IndexType start = _split_work(A.nnnz(), nthreads, tid);
    const IndexType stop  = _split_work(A.nnnz(), nthreads, tid + 1);
    const IndexType last  = stop - 1;

    // multiply value arrays with the corresponding vector values
    // indexed by the colum_indices array;
    for (IndexType n = start; n < stop; n++) {
      out[n] = A.values[n] * x[A.column_indices[n]];
    }

    // perform a segmented scan by key using the row_indices array as key
    Morpheus::inclusive_scan_by_key<Kokkos::Serial>(A.row_indices, out, out,
                                                    stop - start, start);

    const IndexType has_row_stop =
        (tid == nthreads - 1)
            ? 1
            : is_row_stop(A.row_indices, start, last) ||
                  is_row_stop(A.row_indices, start, start + 1);
    IndexType spans_threads =
        (tid == 0) ? ValueType(0)
                   : !is_row_stop(A.row_indices, start - 1, start);

    // adjacent synchronization between threads
    if (spans_threads && !has_row_stop && tid != 0) {
      while (grp_sum[tid - 1] == max_val) {
        // wait for the previous group to update its sum
      };
      grp_sum[tid] = out[last] + grp_sum[tid - 1];
    } else {
      grp_sum[tid] = out[last];
    }

    IndexType is_first_stop = 1;
    for (IndexType n = start; n < stop; n++) {
      IndexType is_stop = (tid == nthreads - 1 && n == last)
                              ? 1
                              : is_row_stop(A.row_indices, n, n + 1);
      if (is_stop) {
        if (is_first_stop) {  // in workgroup
          is_first_stop = 0;
          if (spans_threads) {  // might span multiple threads
            while (grp_sum[tid - 1] == max_val) {
              // wait for the previous group to update its sum
            };
            y[A.row_indices[n]] = out[n] + grp_sum[tid - 1];
          } else {
            y[A.row_indices[n]] = out[n];
          }
        } else {
          y[A.row_indices[n]] = out[n];
        }
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP