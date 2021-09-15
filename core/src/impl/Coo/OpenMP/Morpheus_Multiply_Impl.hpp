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
    const Matrix& A, const Vector& x, Vector& y, CooTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using ValueType = typename Matrix::value_type;
  using IndexType = typename Matrix::index_type;
  using vector    = Vector;

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
    const IndexType spans_workgroups =
        (tid == 0) ? ValueType(0)
                   : !is_row_stop(A.row_indices, start - 1, start);

    // Adjacent synchronization between workgroups:
    // Workgroup 0 updates the first entry ‘grp_sum[0]’ with its last partial
    // sum. For a subsequent workgroup with id X, if it does not contain a row
    // stop, it waits for the entry ‘grp_sum[X-1]’ to be changed from the
    // initial value, i.e., updated by workgroup (X-1), and then updates
    // ‘grp_sum[X]’ with the sum of its last partial sum and ‘grp_sum[X-1]’. If
    // a workgroup contains a row stop, it breaks such chained updates and
    // directly updates ‘grp_sum[X]’ with its last partial sum.
    if (spans_workgroups && !has_row_stop && tid != 0) {
      // wait for the previous group to update its sum
      while (grp_sum[tid - 1] == max_val) {
        // Make sure the read picks up a good copy from memory
#pragma omp flush(grp_sum)
      };
      grp_sum[tid] = out[last] + grp_sum[tid - 1];
      // Make sure other threads can see my write.
#pragma omp flush(grp_sum)
    } else {
      grp_sum[tid] = out[last];
      // Make sure other threads can see my write.
#pragma omp flush(grp_sum)
    }

    IndexType is_first_stop = 1;

    for (IndexType n = start; n < stop; n++) {
      IndexType is_stop = (tid == nthreads - 1 && n == last)
                              ? 1
                              : is_row_stop(A.row_indices, n, n + 1);
      if (is_stop) {
        ValueType R;
        if (is_first_stop) {  // in workgroup
          is_first_stop = 0;
          if (spans_workgroups) {
            // wait for the previous group to update its sum
            while (grp_sum[tid - 1] == max_val) {
              // Make sure the read picks up a good copy from memory
#pragma omp flush(grp_sum)
            };
            R = out[n] + grp_sum[tid - 1];
          } else {
            R = out[n];
          }
        } else {
          R = out[n];
        }

        y[A.row_indices[n]] = R;
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_COO_OPENMP_MULTIPLY_IMPL_HPP