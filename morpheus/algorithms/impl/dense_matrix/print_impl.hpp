/**
 * print_impl.hpp
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_DENSEMATRIX_PRINT_IMPL_HPP
#define MORPHEUS_ALGORITHMS_IMPL_DENSEMATRIX_PRINT_IMPL_HPP

#include <iostream>
#include <iomanip>

#include <morpheus/containers/dense_matrix.hpp>

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, Morpheus::DenseMatrixTag) {
  using index_type = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << "> with " << p.ncols() << " entries\n";

  for (index_type i = 0; i < p.nrows(); i++) {
    for (index_type j = 0; j < p.ncols(); j++) {
      s << " " << std::setw(14) << i;
      s << " " << std::setw(14) << j;
      s << " " << std::setprecision(4) << std::setw(8) << "(" << p(i, j)
        << ")\n";
    }
  }
}
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_DENSEMATRIX_PRINT_IMPL_HPP