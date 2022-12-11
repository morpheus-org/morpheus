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

#ifndef MORPHEUS_COO_PRINT_IMPL_HPP
#define MORPHEUS_COO_PRINT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>
#include <impl/Morpheus_Utils.hpp>

#include <iostream>
#include <iomanip>  // setw, setprecision

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s,
           typename std::enable_if<
               Morpheus::is_coo_matrix_format_container_v<Printable>>::type* =
               nullptr) {
  using index_type = typename Printable::index_type;
  print_matrix_header(p, s);

  for (index_type n = 0; n < p.nnnz(); n++) {
    s << " " << std::setw(14) << p.crow_indices(n);
    s << " " << std::setw(14) << p.ccolumn_indices(n);
    s << " " << std::setprecision(12) << std::setw(12) << "(" << p.cvalues(n)
      << ")\n";
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_COO_PRINT_IMPL_HPP