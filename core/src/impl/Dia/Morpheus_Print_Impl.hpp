/**
 * Morpheus_Print_Impl.hpp
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

#ifndef MORPHEUS_DIA_PRINT_IMPL_HPP
#define MORPHEUS_DIA_PRINT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <impl/Morpheus_Utils.hpp>

#include <iostream>
#include <iomanip>    // setw, setprecision
#include <algorithm>  // max, min

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, DiaTag) {
  print_matrix_header(p, s);

  using IndexType       = typename Printable::index_type;
  using ValueType       = typename Printable::value_type;
  const IndexType ndiag = p.values.ncols();

  for (IndexType i = 0; i < ndiag; i++) {
    const IndexType k = p.diagonal_offsets[i];

    const IndexType i_start = std::max<IndexType>(0, -k);
    const IndexType j_start = std::max<IndexType>(0, k);

    // number of elements to process in this diagonal
    const IndexType N = std::min(p.nrows() - i_start, p.ncols() - j_start);

    for (IndexType n = 0; n < N; n++) {
      ValueType temp = p.values(i_start + n, i);
      if (temp != ValueType(0)) {
        s << " " << std::setw(14) << i;
        s << " " << std::setw(14) << i_start + n;
        s << " " << std::setw(14) << j_start + n;
        s << " " << std::setprecision(4) << std::setw(8) << "(" << temp
          << ")\n";
      }
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_PRINT_IMPL_HPP