/**
 * Morpheus_Utils.hpp
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

#ifndef MORPHEUS_UTILS_HPP
#define MORPHEUS_UTILS_HPP

#include <iostream>

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print_matrix_header(const Printable& p, Stream& s) {
  using IndexType = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";
}

// Calculates padding to align the data based on the current length
template <typename T>
inline const T get_pad_size(T len, T alignment) {
  return alignment * ((len + alignment - 1) / alignment);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_UTILS_HPP