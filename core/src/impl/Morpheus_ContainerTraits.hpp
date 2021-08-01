/**
 * Morpheus_ContainerTraits.hpp
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

#ifndef MORPHEUS_CONTAINERTRAITS_HPP
#define MORPHEUS_CONTAINERTRAITS_HPP

#include <type_traits>

#include <Morpheus_Core.hpp>

namespace Morpheus {

namespace Impl {

/** @class ContainerTraits
 * @brief Traits class for accessing attributes of a Container (Matrix or
 * Vector)
 *
 * Template argument options:
 *  - ContainerTraits<ValueType>
 *  - ContainerTraits<ValueType, ArrayLayout>
 *  - ContainerTraits<ValueType, IndexType, Space>
 *  - ContainerTraits<ValueType, IndexType, ArrayLayout>
 *  - ContainerTraits<ValueType, IndexType, ArrayLayout, Space>
 *  - ContainerTraits<ValueType, ArrayLayout, Space>
 */
template <typename ValueType, class... Properties>
struct ContainerTraits;

template <>
struct ContainerTraits<void> {
  using index_type      = void;
  using array_layout    = void;
  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
};

template <class... Prop>
struct ContainerTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'

  using index_type   = typename ContainerTraits<void, Prop...>::index_type;
  using array_layout = typename ContainerTraits<void, Prop...>::array_layout;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space = typename ContainerTraits<void, Prop...>::memory_space;
  using HostMirrorSpace =
      typename ContainerTraits<void, Prop...>::HostMirrorSpace;
};

template <typename IndexType, class... Prop>
struct ContainerTraits<
    typename std::enable_if_t<std::is_integral<IndexType>::value>, IndexType,
    Prop...> {
  // Specify index type, keep subsequent layout and space arguments

  using index_type   = IndexType;
  using array_layout = typename ContainerTraits<void, Prop...>::array_layout;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space = typename ContainerTraits<void, Prop...>::memory_space;
  using HostMirrorSpace =
      typename ContainerTraits<void, Prop...>::HostMirrorSpace;
};

template <typename ArrayLayout, class... Prop>
struct ContainerTraits<typename std::enable_if_t<
                           Kokkos::Impl::is_array_layout<ArrayLayout>::value>,
                       ArrayLayout, Prop...> {
  // Specify Layout, Space should be the only subsequent argument.

  static_assert(
      std::is_same<typename ContainerTraits<void, Prop...>::array_layout,
                   void>::value,
      "Only one Layout template argument");

  using index_type   = typename ContainerTraits<void, Prop...>::index_type;
  using array_layout = ArrayLayout;
  using execution_space =
      typename ContainerTraits<void, Prop...>::execution_space;
  using memory_space = typename ContainerTraits<void, Prop...>::memory_space;
  using HostMirrorSpace =
      typename ContainerTraits<void, Prop...>::HostMirrorSpace;
};

template <class Space, class... Prop>
struct ContainerTraits<
    typename std::enable_if<Kokkos::Impl::is_space<Space>::value>::type, Space,
    Prop...> {
  // Specify Space, there should not be any other subsequent arguments.

  static_assert(
      std::is_same<typename ContainerTraits<void, Prop...>::index_type,
                   void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::execution_space,
                       void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ContainerTraits<void, Prop...>::array_layout,
                       void>::value,
      "Space is the final optional template argument for a Container");

  using index_type      = typename ContainerTraits<void, Prop...>::index_type;
  using array_layout    = typename ContainerTraits<void, Prop...>::array_layout;
  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using HostMirrorSpace =
      typename Kokkos::Impl::HostMirror<Space>::Space::memory_space;
};

template <class ValueType, class... Properties>
struct ContainerTraits {
 private:
  // Unpack first the properties arguments
  using prop = ContainerTraits<void, Properties...>;

  using IndexType = typename std::conditional_t<
      !std::is_same_v<typename prop::index_type, void>,
      typename prop::index_type, int>;

  using ExecutionSpace = typename std::conditional_t<
      !std::is_same_v<typename prop::execution_space, void>,
      typename prop::execution_space, Kokkos::DefaultExecutionSpace>;

  using MemorySpace = typename std::conditional_t<
      !std::is_same_v<typename prop::memory_space, void>,
      typename prop::memory_space, typename ExecutionSpace::memory_space>;

  using ArrayLayout = typename std::conditional_t<
      !std::is_same_v<typename prop::array_layout, void>,
      typename prop::array_layout, typename ExecutionSpace::array_layout>;

  using HostMirrorSpace = typename std::conditional<
      !std::is_same<typename prop::HostMirrorSpace, void>::value,
      typename prop::HostMirrorSpace,
      typename Kokkos::Impl::HostMirror<ExecutionSpace>::Space>::type;

  // Check the validity of ValueType
  static_assert(std::is_arithmetic_v<ValueType>,
                "ValueType must be an arithmetic type such as int or double");

 public:
  using value_type           = ValueType;
  using const_value_type     = typename std::add_const<ValueType>::type;
  using non_const_value_type = typename std::remove_const<ValueType>::type;

  using index_type   = IndexType;
  using array_layout = ArrayLayout;

  using execution_space   = ExecutionSpace;
  using memory_space      = MemorySpace;
  using device_type       = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using host_mirror_space = HostMirrorSpace;

  enum { is_hostspace = std::is_same<MemorySpace, Kokkos::HostSpace>::value };
};

}  // namespace Impl
}  // namespace Morpheus
#endif  // MORPHEUS_CONTAINERTRAITS_HPP