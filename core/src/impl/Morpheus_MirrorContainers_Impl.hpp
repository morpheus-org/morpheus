/**
 * Morpheus_MirrorContainers_Impl.hpp
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

#ifndef MORPHEUS_MIRRORCONTAINERS_IMPL_HPP
#define MORPHEUS_MIRRORCONTAINERS_IMPL_HPP

#include <type_traits>

namespace Morpheus {
namespace Impl {

template <class Space, template <class, class...> class Container, class T,
          class... P>
struct MirrorContainerType {
  // The incoming container_type
  using src_container_type = Container<T, P...>;
  // The memory space for the mirror container
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space,
                     typename src_container_type::memory_space>::value
  };
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using value_type   = typename src_container_type::non_const_value_type;
  using index_type   = typename src_container_type::index_type;
  using array_layout = typename src_container_type::array_layout;
  // The destination container type if it is not the same memory space - note
  // that the new container will always be managed even when the source is
  // unmanaged
  using dest_container_type =
      Container<value_type, index_type, array_layout, Space>;
  // If it is the same memory_space return the existsing container_type
  using container_type =
      typename std::conditional<is_same_memspace, src_container_type,
                                dest_container_type>::type;
};

template <class Space, template <class, class...> class Container, class T,
          class... P>
struct MirrorType {
  // The incoming container_type
  using src_container_type = Container<T, P...>;
  // The memory space for the mirror container
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space,
                     typename src_container_type::memory_space>::value
  };
  // we want it non-const to allow deep_copy.
  using value_type   = typename src_container_type::non_const_value_type;
  using index_type   = typename src_container_type::non_const_index_type;
  using array_layout = typename src_container_type::array_layout;
  // The destination container type if it is not the same memory space - note
  // that the new container will always be managed even when the source is
  // unmanaged
  using container_type = Container<value_type, index_type, array_layout, Space>;
};

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_MIRRORCONTAINERS_IMPL_HPP