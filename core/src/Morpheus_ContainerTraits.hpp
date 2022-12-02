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

#ifndef MORPHEUS_CONTAINER_TRAITS_HPP
#define MORPHEUS_CONTAINER_TRAITS_HPP

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

  using Backend =
      typename std::conditional_t<!std::is_same_v<typename prop::backend, void>,
                                  typename prop::backend,
                                  Morpheus::DefaultExecutionSpace>;

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
  /*! @brief The type of values held by the container
   */
  using value_type = ValueType;
  /*! @brief The const type of values held by the container
   */
  using const_value_type = typename std::add_const<ValueType>::type;
  /*! @brief The non-const type of values held by the container
   */
  using non_const_value_type = typename std::remove_const<ValueType>::type;
  /*! @brief The type of indices held by the container
   */
  using index_type = IndexType;
  /*! @brief The non-const type of indices held by the container
   */
  using non_const_index_type = typename std::remove_const<IndexType>::type;

  /*! @brief The storage layout of data held by the container
   */
  using array_layout = ArrayLayout;
  /*! @brief The backend out of which algorithms will be dispatched from.
   */
  using backend = Backend;
  /*! @brief The space in which member functions will be executed in.
   */
  using execution_space = ExecutionSpace;
  /*! @brief The space in which data will be stored in.
   */
  using memory_space = MemorySpace;
  /*! @brief A device aware of the execution, memory spaces and backend.
   */
  using device_type = Morpheus::Device<execution_space, memory_space, backend>;
  /*! @brief Represents the user's intended access behaviour.
   */
  using memory_traits = MemoryTraits;
  /*! @brief The host equivalent backend.
   */
  using host_mirror_backend = typename Morpheus::HostMirror<backend>::backend;

  /*! @brief The complete type of the container
   */
  using type =
      Container<value_type, index_type, array_layout, backend, memory_traits>;

  /*! @brief The host mirror equivalent for the container.
   * \note HostMirror is assumed to always be a managed container.
   */
  using HostMirror =
      Container<non_const_value_type, non_const_index_type, array_layout,
                Morpheus::Device<typename host_mirror_backend::execution_space,
                                 typename host_mirror_backend::memory_space,
                                 typename host_mirror_backend::backend>,
                typename Kokkos::MemoryManaged>;

  /*! @brief The pointer type of the container
   */
  using pointer = typename std::add_pointer<type>::type;
  /*! @brief The const pointer type of the container
   */
  using const_pointer =
      typename std::add_pointer<typename std::add_const<type>::type>::type;
  /*! @brief The reference type of the container
   */
  using reference = typename std::add_lvalue_reference<type>::type;
  /*! @brief The const reference type of the container
   */
  using const_reference = typename std::add_lvalue_reference<
      typename std::add_const<type>::type>::type;

  enum { is_hostspace = std::is_same<MemorySpace, Kokkos::HostSpace>::value };
  enum { is_managed = MemoryTraits::is_unmanaged == 0 };
};

}  // namespace Morpheus
#endif  // MORPHEUS_CONTAINER_TRAITS_HPP