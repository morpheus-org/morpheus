/**
 * Morpheus_ContainerTraits.hpp
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

#ifndef MORPHEUS_CONTAINERTRAITS_HPP
#define MORPHEUS_CONTAINERTRAITS_HPP

#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_ContainerTraits_Impl.hpp>

namespace Morpheus {
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
 *  - ContainerTraits<ValueType, IndexType, ArrayLayout, Space, MemoryTraits>
 */
template <template <class, class...> class Container, class ValueType,
          class... Properties>
struct ContainerTraits {
 private:
  // Unpack first the properties arguments
  using prop = Impl::ContainerTraits<void, Properties...>;

  using IndexType = typename std::conditional_t<
      !std::is_same_v<typename prop::index_type, void>,
      typename prop::index_type, int>;

  using ExecutionSpace = typename std::conditional_t<
      !std::is_same_v<typename prop::execution_space, void>,
      typename prop::execution_space, Kokkos::DefaultExecutionSpace>;

  using MemorySpace = typename std::conditional_t<
      !std::is_same_v<typename prop::memory_space, void>,
      typename prop::memory_space, typename ExecutionSpace::memory_space>;

  using Space =
      typename std::conditional_t<!std::is_same_v<typename prop::space, void>,
                                  typename prop::space,
                                  Morpheus::DefaultExecutionSpace>;

  //   using Backend =
  //       typename std::conditional_t<!std::is_same_v<typename prop::backend,
  //       void>,
  //                                   typename prop::backend,
  //                                   Morpheus::CustomBackendTag>;

  using ArrayLayout = typename std::conditional_t<
      !std::is_same_v<typename prop::array_layout, void>,
      typename prop::array_layout, typename ExecutionSpace::array_layout>;

  using MemoryTraits = typename std::conditional<
      !std::is_same<typename prop::memory_traits, void>::value,
      typename prop::memory_traits, typename Kokkos::MemoryManaged>::type;

  // Check the validity of ValueType
  static_assert(std::is_arithmetic_v<ValueType>,
                "ValueType must be an arithmetic type such as int or double");

 public:
  using value_type           = ValueType;
  using const_value_type     = typename std::add_const<ValueType>::type;
  using non_const_value_type = typename std::remove_const<ValueType>::type;

  using index_type           = IndexType;
  using non_const_index_type = typename std::remove_const<IndexType>::type;

  using array_layout = ArrayLayout;

  using space = Space;
  //   using backend         = Backend;
  using execution_space = ExecutionSpace;
  using memory_space    = MemorySpace;
  using device_type   = Morpheus::Device<execution_space, memory_space, Space>;
  using memory_traits = MemoryTraits;
  using host_mirror_space = typename Morpheus::HostMirror<Space>::Space;

  using type =
      Container<value_type, index_type, array_layout, Space, memory_traits>;
  //  will be morpheus::device
  using HostMirror =
      Container<non_const_value_type, non_const_index_type, array_layout,
                Morpheus::Device<typename host_mirror_space::execution_space,
                                 typename host_mirror_space::memory_space,
                                 host_mirror_space>>;

  using pointer = typename std::add_pointer<type>::type;
  using const_pointer =
      typename std::add_pointer<typename std::add_const<type>::type>::type;
  using reference       = typename std::add_lvalue_reference<type>::type;
  using const_reference = typename std::add_lvalue_reference<
      typename std::add_const<type>::type>::type;

  enum { is_hostspace = std::is_same<MemorySpace, Kokkos::HostSpace>::value };
  enum { is_managed = MemoryTraits::is_unmanaged == 0 };
};

}  // namespace Morpheus
#endif  // MORPHEUS_CONTAINERTRAITS_HPP