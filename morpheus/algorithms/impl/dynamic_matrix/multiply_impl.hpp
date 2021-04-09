/**
 * multiply_serial.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DYNAMIC_MATRIX_MULTIPLY_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DYNAMIC_MATRIX_MULTIPLY_IMPL_HPP

#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/containers/vector.hpp>

namespace Morpheus {
// forward decl
template <typename ExecSpace, typename Matrix, typename Vector>
void multiply(const ExecSpace& space, const Matrix& A, const Vector& x,
              Vector& y);

namespace Impl {

struct multiply_fn {
  using result_type = void;

  template <typename ExecSpace, typename Mat, typename Vec>
  result_type operator()(const ExecSpace& space, const Mat& A, const Vec& x,
                         Vec& y) const {
    Morpheus::multiply(space, A, x, y);
  }
};

template <typename ExecSpace, typename Matrix, typename Vector>
void multiply(const ExecSpace& space, const Matrix& A, const Vector& x,
              Vector& y, Morpheus::DynamicTag) {
  std::visit(std::bind(Impl::multiply_fn(), std::cref(space),
                       std::placeholders::_1, std::cref(x), std::ref(y)),
             A.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DYNAMIC_MATRIX_MULTIPLY_IMPL_HPP