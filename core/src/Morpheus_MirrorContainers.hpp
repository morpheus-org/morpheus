/**
 * Morpheus_MirrorContainers.hpp
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

#ifndef MORPHEUS_MIRRORCONTAINERS_HPP
#define MORPHEUS_MIRRORCONTAINERS_HPP

#include <Morpheus_DenseVector.hpp>

namespace Morpheus {

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror(
    const Container<T, P...>& src) {
  using src_type = Container<T, P...>;
  using dst_type = typename src_type::HostMirror;

  return dst_type(src.name().append("Mirror_"), src.nrows(), src.ncols(),
                  src.nnnz());
}

template <class T, class... P>
typename DenseVector<T, P...>::HostMirror create_mirror(
    const DenseVector<T, P...>& src) {
  using src_type = DenseVector<T, P...>;
  using dst_type = typename src_type::HostMirror;

  return dst_type(src.name().append("Mirror_"), src.size());
}

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<
        (std::is_same<
             typename Container<T, P...>::memory_space,
             typename Container<T, P...>::HostMirror::memory_space>::value &&
         std::is_same<typename Container<T, P...>::value_type,
                      typename Container<T, P...>::HostMirror::value_type>::
             value)>::type* = nullptr) {
  return src;
}

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<
        !(std::is_same<
              typename Container<T, P...>::memory_space,
              typename Container<T, P...>::HostMirror::memory_space>::value &&
          std::is_same<typename Container<T, P...>::value_type,
                       typename Container<T, P...>::HostMirror::value_type>::
              value)>::type* = nullptr) {
  return Morpheus::create_mirror(src);
}

}  // namespace Morpheus

#endif  // MORPHEUS_MIRRORCONTAINERS_HPP