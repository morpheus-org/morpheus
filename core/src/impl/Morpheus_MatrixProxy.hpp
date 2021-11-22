/**
 * Morpheus_MatrixProxy.hpp
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

#ifndef MORPHEUS_MATRIXPROXY_HPP
#define MORPHEUS_MATRIXPROXY_HPP

#include <impl/Morpheus_Variant.hpp>

#include <tuple>

namespace Morpheus {
// Compile-time type list with indexed access
template <class... Args>
struct TypeList {
  template <std::size_t N>
  using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
};

template <class... Formats>
struct MatrixFormatsProxy {
  using type = MatrixFormatsProxy<Formats...>;

  using variant   = Morpheus::Impl::Variant::variant<Formats...>;
  using type_list = TypeList<Formats...>;
};

}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXPROXY_HPP