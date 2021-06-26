/**
 * multiply_impl.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_OMP_MULTIPLY_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_OMP_MULTIPLY_IMPL_HPP

#include <morpheus/core/macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <morpheus/core/type_traits.hpp>
#include <morpheus/core/exceptions.hpp>
#include <morpheus/containers/impl/format_tags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(
    const ExecSpace& space, const LinearOperator& A, const MatrixOrVector1& x,
    MatrixOrVector2& y, DiaTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using I = typename LinearOperator::index_type;

#pragma omp parallel for
  for (I i = 0; i < (int)A.diagonal_offsets.size(); i++) {
    const I k       = A.diagonal_offsets[i];  // diagonal offset
    const I i_start = std::max(0, -k);
    const I j_start = std::max(0, k);
    const I N       = std::min(A.nrows() - i_start, A.ncols() - j_start);

    for (I n = 0; n < N; n++) {
      y[i_start + n] += A.values(i, j_start + n) * x[j_start + n];
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ALGORITHMS_IMPL_DIA_MATRIX_OMP_MULTIPLY_IMPL_HPP