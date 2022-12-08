/**
 * Morpheus_Dot.hpp
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

#ifndef MORPHEUS_DOT_HPP
#define MORPHEUS_DOT_HPP

#include <impl/Morpheus_Dot_Impl.hpp>

namespace Morpheus {

template <typename ExecSpace, typename Vector1, typename Vector2>
inline typename Vector2::value_type dot(typename Vector1::index_type n,
                                        const Vector1& x, const Vector2& y) {
  static_assert(is_format_compatible<Vector1, Vector2>::value,
                "x and y must be compatible types");
  return Impl::dot<ExecSpace>(n, x, y);
}

}  // namespace Morpheus

#endif  // MORPHEUS_DOT_HPP