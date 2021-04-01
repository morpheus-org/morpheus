/**
 * multiply_impl_openmp.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_OPENMP_HPP
#define MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_OPENMP_HPP

#include <morpheus/core/macros.hpp>
#if defined(MORPHEUS_ENABLE_OPENMP)

#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/vector.hpp>
#include <morpheus/core/exceptions.hpp>
namespace Morpheus {
namespace Impl {

template <typename Matrix, typename Vector>
void multiply(const Matrix& A, const Vector& x, Vector& y, Morpheus::CsrTag,
              typename std::enable_if<
                  std::is_same<typename Matrix::execution_space,
                               Kokkos::OpenMP::execution_space>::value,
                  Kokkos::OpenMP::execution_space>::type* = nullptr) {
  throw Morpheus::NotImplementedException(
      "void multiply(const " + A.name() + "& A, const " + x.name() + "& x, " +
      y.name() + "& y," + "Morpheus::CsrTag, Kokkos::OpenMP)");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ALGORITHMS_IMPL_CSR_MATRIX_MULTIPLY_IMPL_OPENMP_HPP