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

#ifndef MORPHEUS_CSR_PRINT_IMPL_HPP
#define MORPHEUS_CSR_PRINT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <impl/Morpheus_Utils.hpp>

#include <iostream>
#include <iomanip>  // setw, setprecision

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s,
           typename std::enable_if<
               Morpheus::is_csr_matrix_format_container_v<Printable>>::type* =
               nullptr) {
  using size_type  = typename Printable::size_type;
  using index_type = typename Printable::index_type;
  print_matrix_header(p, s);

  for (size_type i = 0; i < p.nrows(); i++) {
    for (index_type jj = p.crow_offsets(i); jj < p.crow_offsets(i + 1); jj++) {
      s << " " << std::setw(14) << i;
      s << " " << std::setw(14) << p.ccolumn_indices(jj);
      s << " " << std::setprecision(4) << std::setw(8) << "(" << p.cvalues(jj)
        << ")\n";
    }
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CSR_PRINT_IMPL_HPP