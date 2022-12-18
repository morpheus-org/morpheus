/**
 * Morpheus_Print_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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
void print(const Printable& p, Stream& s,
           typename std::enable_if<
               Morpheus::is_dia_matrix_format_container_v<Printable>>::type* =
               nullptr) {
  using size_type       = typename Printable::size_type;
  using index_type      = typename Printable::index_type;
  using value_type      = typename Printable::value_type;
  const size_type ndiag = p.cvalues().ncols();

  print_matrix_header(p, s);

  for (size_type i = 0; i < ndiag; i++) {
    const index_type k = p.cdiagonal_offsets(i);

    const size_type i_start = std::max<size_type>(0, -k);
    const size_type j_start = std::max<size_type>(0, k);

    // number of elements to process in this diagonal
    const size_type N = std::min(p.nrows() - i_start, p.ncols() - j_start);

    for (size_type n = 0; n < N; n++) {
      value_type temp = p.cvalues(i_start + n, i);
      if (temp != value_type(0)) {
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