/**
 * Morpheus_MatrixOperations_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>
#include <fwd/Morpheus_Fwd_Algorithms.hpp>

#include <variant>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Algorithm, typename SparseMatrix,
          typename Vector>
inline void update_diagonal(SparseMatrix& A, const Vector& diagonal,
                            Morpheus::DynamicTag, Morpheus::DenseVectorTag,
                            Algorithm) {
  std::visit(
      [&](auto&& arg) {
        Morpheus::update_diagonal<ExecSpace>(arg, diagonal, Algorithm{});
      },
      A.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_MATRIXOPERATIONS_IMPL_HPP