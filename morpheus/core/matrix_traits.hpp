/**
 * matrix_traits.hpp
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

#ifndef MORPHEUS_CORE_MATRIX_TRAITS_HPP
#define MORPHEUS_CORE_MATRIX_TRAITS_HPP

#include <type_traits>
#include <morpheus/core/core.hpp>

namespace Morpheus {

namespace Impl {

/** @class MatrixTraits
 * @brief Traits class for accessing attributes of a Matrix
 *
 * Template argument options:
 *  - MatrixTraits<ValueType>
 *  - MatrixTraits<ValueType, ArrayLayout>
 *  - MatrixTraits<ValueType, IndexType, Space>
 *  - MatrixTraits<ValueType, IndexType, ArrayLayout>
 *  - MatrixTraits<ValueType, IndexType, ArrayLayout, Space>
 *  - MatrixTraits<ValueType, ArrayLayout, Space>
 */
template <typename ValueType, class... Properties>
struct MatrixTraits;

template <>
struct MatrixTraits<void> {
  using index_type      = void;
  using array_layout    = void;
  using execution_space = void;
  using memory_space    = void;
};

template <class... Prop>
struct MatrixTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'

  using index_type      = typename MatrixTraits<void, Prop...>::index_type;
  using array_layout    = typename MatrixTraits<void, Prop...>::array_layout;
  using execution_space = typename MatrixTraits<void, Prop...>::execution_space;
  using memory_space    = typename MatrixTraits<void, Prop...>::memory_space;
};

template <typename IndexType, class... Prop>
struct MatrixTraits<
    typename std::enable_if_t<std::is_integral<IndexType>::value>, IndexType,
    Prop...> {
  // Specify index type, keep subsequent layout and space arguments

  using index_type      = IndexType;
  using array_layout    = typename MatrixTraits<void, Prop...>::array_layout;
  using execution_space = typename MatrixTraits<void, Prop...>::execution_space;
  using memory_space    = typename MatrixTraits<void, Prop...>::memory_space;
};

template <typename ArrayLayout, class... Prop>
struct MatrixTraits<typename std::enable_if_t<
                        Kokkos::Impl::is_array_layout<ArrayLayout>::value>,
                    ArrayLayout, Prop...> {
  // Specify Layout, keep subsequent space arguments

  using index_type      = void;
  using array_layout    = ArrayLayout;
  using execution_space = typename MatrixTraits<void, Prop...>::execution_space;
  using memory_space    = typename MatrixTraits<void, Prop...>::memory_space;
};

template <class Space, class... Prop>
struct MatrixTraits<
    typename std::enable_if<Kokkos::Impl::is_space<Space>::value>::type, Space,
    Prop...> {
  // Specify Space, there should not be any other subsequent arguments.

  static_assert(
      std::is_same_v<typename MatrixTraits<void, Prop...>::execution_space,
                     void> &&
          std::is_same_v<typename MatrixTraits<void, Prop...>::memory_space,
                         void>,
      "Only one Matrix Execution or Memory Space template argument");

  using index_type      = void;
  using array_layout    = void;
  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
};

template <class ValueType, class... Properties>
struct MatrixTraits {
 private:
  // Unpack first the properties arguments
  using prop = MatrixTraits<void, Properties...>;

  using IndexType = typename std::conditional_t<
      !std::is_same_v<typename prop::index_type, void>,
      typename prop::index_type, int>;

  using ExecutionSpace = typename std::conditional_t<
      !std::is_same_v<typename prop::execution_space, void>,
      typename prop::execution_space, Kokkos::DefaultExecutionSpace>;

  using ArrayLayout = typename std::conditional_t<
      !std::is_same_v<typename prop::array_layout, void>,
      typename prop::array_layout, typename ExecutionSpace::array_layout>;

  using MemorySpace = typename std::conditional_t<
      !std::is_same_v<typename prop::memory_space, void>,
      typename prop::memory_space, typename ExecutionSpace::memory_space>;

  // Check the validity of ValueType
  static_assert(std::is_arithmetic_v<ValueType>,
                "ValueType must be an arithmetic type such as int or double");

 public:
  using value_type   = ValueType;
  using index_type   = IndexType;
  using array_layout = ArrayLayout;

  using execution_space = ExecutionSpace;
  using memory_space    = MemorySpace;
  using device_type     = Kokkos::Device<ExecutionSpace, MemorySpace>;
};
}  // namespace Impl
}  // namespace Morpheus
#endif  // MORPHEUS_CORE_MATRIX_TRAITS_HPP