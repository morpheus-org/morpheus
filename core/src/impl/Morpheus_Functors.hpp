/**
 * Morpheus_Functors.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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
#ifndef MORPHEUS_FUNCTORS_HPP
#define MORPHEUS_FUNCTORS_HPP

#include <Kokkos_Core.hpp>

namespace Morpheus {
namespace Impl {
// primary template
template <typename View, typename... Types>
struct set_functor {};

template <typename View, typename ValueType, typename IndexType>
struct set_functor<View, ValueType, IndexType> {
  View _data;
  ValueType _val;
  IndexType _ncols;

  set_functor(View data, ValueType val, IndexType ncols)
      : _data(data), _val(val), _ncols(ncols) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType& i) const {
    for (IndexType j = 0; j < _ncols; j++) {
      _data(i, j) = _val;
    }
  }
};

template <typename View, typename ValueType>
struct set_functor<View, ValueType> {
  View _data;
  ValueType _val;

  set_functor(View data, ValueType val) : _data(data), _val(val) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const { _data(i) = _val; }
};

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_FUNCTORS_HPP