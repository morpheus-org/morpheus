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
#include <Morpheus_AlgorithmTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, DiaTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_OpenMP_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using IndexType       = typename Matrix::index_type;
  using ValueType       = typename Matrix::value_type;
  const IndexType ndiag = A.values.ncols();

#pragma omp parallel for
  for (IndexType row = 0; row < A.nrows(); row++) {
    ValueType sum = ValueType(0);

    for (IndexType n = 0; n < ndiag; n++) {
      const IndexType col = row + A.diagonal_offsets[n];

      if (col >= 0 && col < A.ncols()) {
        sum += A.values(row, n) * x[col];
      }
    }
    y[row] = sum;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_DIA_OPENMP_MULTIPLY_IMPL_HPP