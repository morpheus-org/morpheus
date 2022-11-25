/**
 * Morpheus_ContainerTraits_Impl.hpp
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

#ifndef MORPHEUS_CONTAINERTRAITS_IMPL_HPP
#define MORPHEUS_CONTAINERTRAITS_IMPL_HPP

#include <Morpheus_GenericBackend.hpp>

#include <Kokkos_Core.hpp>

namespace Morpheus {

namespace Impl {

template <typename ValueType, class... Properties>
struct ContainerTraits;

template <>
struct ContainerTraits<void> {
  using index_type      = void;
  using array_layout    = void;
  using backend         = void;
  using execution_space = void;
  using memory_space    = void;
  using memory_traits   = void;
};

template <class... Prop>
struct ContainerTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'

  using index_type   = typename ContainerTraits<void, Prop...>::index_type;
  using array_layout = typename ContainerTraits<void, Prop...>::array_layout;
  using backend      = typename ContainerTraits<void, Prop...>::backend;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space  = typename ContainerTraits<void, Prop...>::memory_space;
  using memory_traits = typename ContainerTraits<void, Prop...>::memory_traits;
};

template <typename IndexType, class... Prop>
struct ContainerTraits<
    typename std::enable_if_t<std::is_integral<IndexType>::value>, IndexType,
    Prop...> {
  // Specify index type
  // Keep subsequent layout, space and memory trait arguments

  using index_type   = IndexType;
  using array_layout = typename ContainerTraits<void, Prop...>::array_layout;
  using backend      = typename ContainerTraits<void, Prop...>::backend;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space  = typename ContainerTraits<void, Prop...>::memory_space;
  using memory_traits = typename ContainerTraits<void, Prop...>::memory_traits;
};

template <typename ArrayLayout, class... Prop>
struct ContainerTraits<typename std::enable_if_t<
                           Kokkos::Impl::is_array_layout<ArrayLayout>::value>,
                       ArrayLayout, Prop...> {
  // Specify Layout
  // Keep Space and MemoryTraits arguments

  using index_type   = void;
  using array_layout = ArrayLayout;
  using backend      = typename ContainerTraits<void, Prop...>::backend;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space  = typename ContainerTraits<void, Prop...>::memory_space;
  using memory_traits = typename ContainerTraits<void, Prop...>::memory_traits;
};

template <class Space, class... Prop>
struct ContainerTraits<
    typename std::enable_if<Morpheus::is_space<Space>::value>::type, Space,
    Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
      std::is_same<typename ContainerTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::array_layout,
                       void>::value,
      "Only one Container Execution or Memory Space template argument");

  using index_type   = void;
  using array_layout = void;
  using backend      = typename std::conditional<
      !Morpheus::has_backend<Space>::value,  // means we are using
                                             // Kokkos::<space>
      Morpheus::GenericBackend<Space>, Space>::type;
  using execution_space = typename backend::execution_space;
  using memory_space    = typename backend::memory_space;
  using memory_traits = typename ContainerTraits<void, Prop...>::memory_traits;
};

template <class MemoryTraits, class... Prop>
struct ContainerTraits<typename std::enable_if<
                           Kokkos::is_memory_traits<MemoryTraits>::value>::type,
                       MemoryTraits, Prop...> {
  // Specify memory trait, should not be any subsequent arguments

  static_assert(
      std::is_same<typename ContainerTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::array_layout,
                       void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::memory_traits,
                       void>::value,
      "MemoryTrait is the final optional template argument for a Container");

  using index_type      = void;
  using backend         = void;
  using execution_space = void;
  using memory_space    = void;
  using array_layout    = void;
  using memory_traits   = MemoryTraits;
};

}  // namespace Impl
}  // namespace Morpheus
#endif  // MORPHEUS_CONTAINERTRAITS_IMPL_HPP