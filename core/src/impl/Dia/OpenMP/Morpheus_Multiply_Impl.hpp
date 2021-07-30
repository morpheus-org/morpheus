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

#ifndef MORPHEUS_DIA_OPENMP_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_OPENMP_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    DiaTag, DenseVectorTag, DenseVectorTag,
    typename std::enable_if_t<
        Morpheus::is_execution_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, LinearOperator> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector1> &&
        Morpheus::has_access_v<ExecSpace, MatrixOrVector2>>* = nullptr) {
  using IndexType       = typename LinearOperator::index_type;
  const IndexType ndiag = A.values.ncols();

  for (IndexType i = 0; i < A.nrows(); i++) y[i] = 0;

#pragma omp parallel for
  for (IndexType i = 0; i < ndiag; i++) {
    const IndexType k = A.diagonal_offsets[i];

    const IndexType i_start = std::max<IndexType>(0, -k);
    const IndexType j_start = std::max<IndexType>(0, k);

    // number of elements to process in this diagonal
    const IndexType N = std::min(A.nrows() - i_start, A.ncols() - j_start);

    for (IndexType n = 0; n < N; n++) {
      y[i_start + n] += A.values(i_start + n, i) * x[j_start + n];
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DIA_OPENMP_MULTIPLY_IMPL_HPP