/**
 * Morpheus_Print.hpp
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

#ifndef MORPHEUS_PRINT_HPP
#define MORPHEUS_PRINT_HPP

#include <impl/Morpheus_Print_Impl.hpp>
#include <impl/Dynamic/Morpheus_Print_Impl.hpp>

namespace Morpheus {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s) {
  Impl::print(p, s);
}

template <typename Printable>
void print(const Printable& p) {
  Morpheus::print(p, std::cout);
}

}  // namespace Morpheus

#endif  // MORPHEUS_PRINT_HPP