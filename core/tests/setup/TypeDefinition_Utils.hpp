/**
 * TestTypeDefinition_Utils.hpp
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

#ifndef MORPHEUS_CORE_TESTS_TYPEDEFINITION_UTILS_HPP
#define MORPHEUS_CORE_TESTS_TYPEDEFINITION_UTILS_HPP

#include <gtest/gtest.h>
#include <Morpheus_Core.hpp>

/*
 * Each container can have any of the following template arguments
 *  - ContainerTraits<ValueType>
 *  - ContainerTraits<ValueType, ArrayLayout>
 *  - ContainerTraits<ValueType, IndexType, Space>
 *  - ContainerTraits<ValueType, IndexType, ArrayLayout>
 *  - ContainerTraits<ValueType, IndexType, ArrayLayout, Space>
 *  - ContainerTraits<ValueType, ArrayLayout, Space>
 *
 */
namespace Impl {
// wrapper that carries out static checks for each containers traits
template <typename Container, typename RefValueType, typename RefIndexType,
          typename RefSpace, typename RefArrayLayout>
struct ContainerTester {
  // Type Traits
  static_assert(
      std::is_same<typename Container::type::value_type, RefValueType>::value);
  static_assert(
      std::is_same<typename Container::type::index_type, RefIndexType>::value);
  static_assert(std::is_same<typename Container::type::array_layout,
                             RefArrayLayout>::value);
  static_assert(std::is_same<typename Container::type::memory_space,
                             typename RefSpace::memory_space>::value);

  // Value Traits
  static_assert(
      std::is_same<typename Container::value_type, RefValueType>::value);
  static_assert(std::is_same<typename Container::non_const_value_type,
                             RefValueType>::value);
  static_assert(
      std::is_same<typename Container::index_type, RefIndexType>::value);
  static_assert(std::is_same<typename Container::non_const_index_type,
                             RefIndexType>::value);

  static_assert(
      std::is_same<typename Container::array_layout, RefArrayLayout>::value);

  // Space Traits
  static_assert(std::is_same<typename Container::memory_space,
                             typename RefSpace::memory_space>::value);
  static_assert(std::is_same<typename Container::execution_space,
                             typename RefSpace::execution_space>::value);
  static_assert(
      std::is_same<typename Container::device_type,
                   Kokkos::Device<typename Container::execution_space,
                                  typename Container::memory_space>>::value);

  //  HostMirror Traits
  static_assert(std::is_same<typename Container::HostMirror::value_type,
                             typename Container::non_const_value_type>::value);
  static_assert(std::is_same<typename Container::HostMirror::index_type,
                             typename Container::non_const_index_type>::value);
  static_assert(std::is_same<typename Container::HostMirror::array_layout,
                             typename Container::array_layout>::value);
  static_assert(
      std::is_same<typename Container::HostMirror::execution_space,
                   Kokkos::DefaultHostExecutionSpace::execution_space>::value);
  static_assert(
      std::is_same<
          typename Container::HostMirror::memory_space,
          typename Kokkos::DefaultHostExecutionSpace::memory_space>::value);

  // Pointer Traits
  static_assert(
      std::is_same<
          typename Container::pointer,
          typename std::add_pointer<typename Container::type>::type>::value);
  static_assert(
      std::is_same<typename Container::const_pointer,
                   typename std::add_pointer<typename std::add_const<
                       typename Container::type>::type>::type>::value);

  //   Reference Traits
  static_assert(std::is_same<typename Container::reference,
                             typename std::add_lvalue_reference<
                                 typename Container::type>::type>::value);
  static_assert(
      std::is_same<typename Container::const_reference,
                   typename std::add_lvalue_reference<typename std::add_const<
                       typename Container::type>::type>::type>::value);

  using type = Container;
};
}  // namespace Impl

#endif  // MORPHEUS_CORE_TESTS_TYPEDEFINITION_UTILS_HPP