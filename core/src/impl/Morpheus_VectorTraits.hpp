/**
 * Morpheus_VectorTraits.hpp
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
#ifndef MORPHEUS_VECTORTRAITS_HPP
#define MORPHEUS_VECTORTRAITS_HPP

#include <type_traits>

#include <Morpheus_Core.hpp>

namespace Morpheus {

namespace Impl {

/** @class VectorTraits
 * @brief Traits class for accessing attributes of a Vector
 *
 * Template argument options:
 *  - VectorTraits<ValueType>
 *  - VectorTraits<ValueType, Space>
 */
template <typename ValueType, class... Properties>
struct VectorTraits;

template <>
struct VectorTraits<void> {
  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
};

template <class... Prop>
struct VectorTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'

  using execution_space = typename VectorTraits<void, Prop...>::execution_space;
  using memory_space    = typename VectorTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename VectorTraits<void, Prop...>::HostMirrorSpace;
};

template <class Space, class... Prop>
struct VectorTraits<
    typename std::enable_if<Kokkos::Impl::is_space<Space>::value>::type, Space,
    Prop...> {
  // Specify Space, there should not be any other subsequent arguments.

  static_assert(
      std::is_same_v<typename VectorTraits<void, Prop...>::execution_space,
                     void> &&
          std::is_same_v<typename VectorTraits<void, Prop...>::memory_space,
                         void>,
      "Only one Vector Execution or Memory Space template argument");
  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using HostMirrorSpace =
      typename Kokkos::Impl::HostMirror<Space>::Space::memory_space;
};

template <class ValueType, class... Properties>
struct VectorTraits {
 private:
  // Unpack first the properties arguments
  using prop = VectorTraits<void, Properties...>;

  using ExecutionSpace = typename std::conditional_t<
      !std::is_same_v<typename prop::execution_space, void>,
      typename prop::execution_space, Kokkos::DefaultExecutionSpace>;

  using MemorySpace = typename std::conditional_t<
      !std::is_same_v<typename prop::memory_space, void>,
      typename prop::memory_space, typename ExecutionSpace::memory_space>;

  using HostMirrorSpace = typename std::conditional<
      !std::is_same<typename prop::HostMirrorSpace, void>::value,
      typename prop::HostMirrorSpace,
      typename Kokkos::Impl::HostMirror<ExecutionSpace>::Space>::type;

  // Check the validity of ValueType
  static_assert(std::is_arithmetic_v<ValueType>,
                "ValueType must be an arithmetic type such as int or double");

 public:
  using value_type = ValueType;

  using execution_space   = ExecutionSpace;
  using memory_space      = MemorySpace;
  using device_type       = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using host_mirror_space = HostMirrorSpace;
};
}  // namespace Impl
}  // namespace Morpheus
#endif  // MORPHEUS_VECTORTRAITS_HPP