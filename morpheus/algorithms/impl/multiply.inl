/**
 * multiply.inl
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_INL
#define MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_INL

#include <morpheus/core/exceptions.hpp>
#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/containers/vector.hpp>

namespace Morpheus {

namespace Impl {

template <typename Matrix, typename Vector>
void multiply(const Matrix& A, const Vector& x, Vector& y, Morpheus::CooTag) {
  using index_type = typename Matrix::index_type;
  for (index_type n = 0; n < A.nnnz(); n++) {
    y[A.row_indices[n]] += A.values[n] * x[A.column_indices[n]];
  }
}

template <typename Matrix, typename Vector>
void multiply(const Matrix& A, const Vector& x, Vector& y,
              Morpheus::Impl::SparseMatTag) {
  Morpheus::NotImplementedException(
      "void multiply(const " + A.name() + "& A, const " + x.name() + "& x, " +
      y.name() + "& y," + "Morpheus::Impl::SparseMatTag)");
}

struct multiply_fn {
  using result_type = void;

  template <typename Mat, typename Vec>
  result_type operator()(const Mat& A, const Vec& x, Vec& y) const {
    Morpheus::multiply(A, x, y);
  }
};

template <typename Matrix, typename Vector>
void multiply(const Matrix& A, const Vector& x, Vector& y,
              Morpheus::DynamicTag) {
  std::visit(std::bind(Impl::multiply_fn(), std::placeholders::_1, std::cref(x),
                       std::ref(y)),
             A.formats());
}

}  // namespace Impl

template <typename Matrix, typename Vector>
void multiply(Matrix const& A, Vector const& x, Vector& y) {
  Impl::multiply(A, x, y, typename Matrix::tag());
}

}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_INL