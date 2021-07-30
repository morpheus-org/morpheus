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

#ifndef MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_IMPL_HPP

// TODO: Let Cmake autogenerate those
#include <impl/Coo/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_Multiply_Impl.hpp>

#include <impl/Csr/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Csr/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Csr/Cuda/Morpheus_Multiply_Impl.hpp>

#include <impl/Dia/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Dia/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Dia/Cuda/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
// forward decl
template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(const LinearOperator& A,
                                       const MatrixOrVector1& x,
                                       MatrixOrVector2& y);

namespace Impl {

template <typename ExecSpace>
struct multiply_fn {
  using result_type = void;

  template <typename LinearOperator, typename MatOrVec1, typename MatrOrVec2>
  MORPHEUS_INLINE_FUNCTION result_type operator()(const LinearOperator& A,
                                                  const MatOrVec1& x,
                                                  MatrOrVec2& y) const {
    Morpheus::multiply<ExecSpace>(A, x, y);
  }
};

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
MORPHEUS_INLINE_FUNCTION void multiply(const LinearOperator& A,
                                       const MatrixOrVector1& x,
                                       MatrixOrVector2& y, Morpheus::DynamicTag,
                                       Morpheus::DenseVectorTag,
                                       Morpheus::DenseVectorTag) {
  std::visit(std::bind(Impl::multiply_fn<ExecSpace>(), std::placeholders::_1,
                       std::cref(x), std::ref(y)),
             A.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_MULTIPLY_IMPL_HPP