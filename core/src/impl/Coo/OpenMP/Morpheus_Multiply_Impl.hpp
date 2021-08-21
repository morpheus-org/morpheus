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

int threads() {
  int t = 1;
#pragma omp parallel
  { t = omp_get_num_threads(); }

  return t;
}

template <typename T>
int is_row_stop(T container, typename T::index_type idx,
                typename T::index_type len = 0) {
  return container[idx] != container[idx + len + 1];
}

template <typename T>
int is_first_row_stop_in_workgroup(T container,
                                   typename T::index_type start_idx,
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

  const IndexType nthreads = threads();
  vector out("out", A.nnnz(), 0), last_partial_sums(nthreads, 0);
  // Initialize to special value
  vector grp_sum(nthreads,
                 std::numeric_limits<typename vector::value_type>::max());

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

    // Check if last value is at a row stop and write partial_sum_out
    if (tid == nthreads - 1) {
      // Always row stop at the end of array
      last_partial_sums[tid] = ValueType(0);
    } else {
      if (is_row_stop(A.row_indices, last))
        last_partial_sums[tid] = ValueType(0);  // Zero at row stop
      else
        last_partial_sums[tid] = out[last];
    }

#pragma omp barrier  // partial_sum_out and out must be available to all

    ValueType partial_sum =
        (tid == 0) ? ValueType(0) : last_partial_sums[tid - 1];

    for (IndexType n = start; n < last; n++) {
      if(is_row_stop(A.row_indices,n){
        if (is_first_row_stop_in_workgroup(A.row_indices, start, n)) {
          // TODO: A row might span multiple threads, need to get the partial
          // sum from those too.
          y[A.row_indices[n]] = out[n] + partial_sum;
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