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
          typename RefArrayLayout, typename RefSpace>
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

// ContainerTraits<ValueType>
template <typename ValueType>
struct MorpheusContainers_v {
  using RefValueType   = ValueType;
  using RefIndexType   = int;                              // Default
  using RefSpace       = Kokkos::DefaultExecutionSpace;    // Default
  using RefArrayLayout = typename RefSpace::array_layout;  // Default

  using DenseVector =
      typename Impl::ContainerTester<Morpheus::DenseVector<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;

  using DenseMatrix =
      typename Impl::ContainerTester<Morpheus::DenseMatrix<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;

  using CooMatrix =
      typename Impl::ContainerTester<Morpheus::CooMatrix<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;

  using CsrMatrix =
      typename Impl::ContainerTester<Morpheus::CsrMatrix<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;

  using DiaMatrix =
      typename Impl::ContainerTester<Morpheus::DiaMatrix<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;

  using DynamicMatrix =
      typename Impl::ContainerTester<Morpheus::DynamicMatrix<ValueType>,
                                     RefValueType, RefIndexType, RefSpace,
                                     RefArrayLayout>::type;
};

// ContainerTraits<ValueType, ArrayLayout>
template <typename ValueType, typename ArrayLayout>
struct MorpheusContainers_vl {
  using RefValueType   = ValueType;
  using RefIndexType   = int;                            // Default
  using RefSpace       = Kokkos::DefaultExecutionSpace;  // Default
  using RefArrayLayout = ArrayLayout;

  using DenseVector = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, ArrayLayout>, RefValueType, RefIndexType,
      RefSpace, RefArrayLayout>::type;

  using DenseMatrix = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, ArrayLayout>, RefValueType, RefIndexType,
      RefSpace, RefArrayLayout>::type;

  using CooMatrix = typename Impl::ContainerTester<
      Morpheus::CooMatrix<ValueType, ArrayLayout>, RefValueType, RefIndexType,
      RefSpace, RefArrayLayout>::type;

  using CsrMatrix = typename Impl::ContainerTester<
      Morpheus::CsrMatrix<ValueType, ArrayLayout>, RefValueType, RefIndexType,
      RefSpace, RefArrayLayout>::type;

  using DiaMatrix = typename Impl::ContainerTester<
      Morpheus::DiaMatrix<ValueType, ArrayLayout>, RefValueType, RefIndexType,
      RefSpace, RefArrayLayout>::type;

  using DynamicMatrix = typename Impl::ContainerTester<
      Morpheus::DynamicMatrix<ValueType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;
};

// ContainerTraits<ValueType, IndexType, Space>
template <typename ValueType, typename IndexType, typename Space>
struct MorpheusContainers_vis {
  using RefValueType   = ValueType;
  using RefIndexType   = IndexType;
  using RefSpace       = Space;
  using RefArrayLayout = typename RefSpace::array_layout;

  using DenseVector = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DenseMatrix = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CooMatrix = typename Impl::ContainerTester<
      Morpheus::CooMatrix<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CsrMatrix = typename Impl::ContainerTester<
      Morpheus::CsrMatrix<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DiaMatrix = typename Impl::ContainerTester<
      Morpheus::DiaMatrix<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DynamicMatrix = typename Impl::ContainerTester<
      Morpheus::DynamicMatrix<ValueType, IndexType, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;
};

// ContainerTraits<ValueType, IndexType, ArrayLayout>
template <typename ValueType, typename IndexType, typename ArrayLayout>
struct MorpheusContainers_vil {
  using RefValueType   = ValueType;
  using RefIndexType   = IndexType;
  using RefSpace       = Kokkos::DefaultExecutionSpace;  // Default
  using RefArrayLayout = ArrayLayout;

  using DenseVector = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DenseMatrix = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CooMatrix = typename Impl::ContainerTester<
      Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CsrMatrix = typename Impl::ContainerTester<
      Morpheus::CsrMatrix<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DiaMatrix = typename Impl::ContainerTester<
      Morpheus::DiaMatrix<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DynamicMatrix = typename Impl::ContainerTester<
      Morpheus::DynamicMatrix<ValueType, IndexType, ArrayLayout>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;
};

// ContainerTraits<ValueType, IndexType, ArrayLayout, Space>
template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct MorpheusContainers_vils {
  using RefValueType   = ValueType;
  using RefIndexType   = IndexType;
  using RefSpace       = Space;
  using RefArrayLayout = ArrayLayout;

  using DenseVector = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;

  using DenseMatrix = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;

  using CooMatrix = typename Impl::ContainerTester<
      Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;

  using CsrMatrix = typename Impl::ContainerTester<
      Morpheus::CsrMatrix<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;

  using DiaMatrix = typename Impl::ContainerTester<
      Morpheus::DiaMatrix<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;

  using DynamicMatrix = typename Impl::ContainerTester<
      Morpheus::DynamicMatrix<ValueType, IndexType, ArrayLayout, Space>,
      RefValueType, RefIndexType, RefSpace, RefArrayLayout>::type;
};

// ContainerTraits<ValueType, ArrayLayout, Space>
template <typename ValueType, typename ArrayLayout, typename Space>
struct MorpheusContainers_vls {
  using RefValueType   = ValueType;
  using RefIndexType   = int;  // Default
  using RefSpace       = Space;
  using RefArrayLayout = ArrayLayout;

  using DenseVector = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DenseMatrix = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CooMatrix = typename Impl::ContainerTester<
      Morpheus::CooMatrix<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using CsrMatrix = typename Impl::ContainerTester<
      Morpheus::CsrMatrix<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DiaMatrix = typename Impl::ContainerTester<
      Morpheus::DiaMatrix<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;

  using DynamicMatrix = typename Impl::ContainerTester<
      Morpheus::DynamicMatrix<ValueType, ArrayLayout, Space>, RefValueType,
      RefIndexType, RefSpace, RefArrayLayout>::type;
};

using ContainerImplementations = ::testing::Types<
    MorpheusContainers_v<double>,
    MorpheusContainers_vl<double, Kokkos::LayoutRight>,
    MorpheusContainers_vis<double, int, Kokkos::Serial>,
    MorpheusContainers_vil<double, int, Kokkos::LayoutRight>,
    MorpheusContainers_vils<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    MorpheusContainers_vls<double, Kokkos::LayoutRight, Kokkos::Serial>>;

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct DenseVectorTypes {
  using DefaultValueType   = double;
  using DefaultIndexType   = int;
  using DefaultSpace       = Kokkos::DefaultExecutionSpace;
  using DefaultArrayLayout = typename DefaultSpace::array_layout;

  using DenseVector_v =
      typename Impl::ContainerTester<Morpheus::DenseVector<ValueType>,
                                     ValueType, DefaultIndexType, DefaultSpace,
                                     DefaultArrayLayout>::type;

  using DenseVector_vl = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, ArrayLayout>, ValueType,
      DefaultIndexType, DefaultSpace, ArrayLayout>::type;

  using DenseVector_vis = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, Space>, ValueType, IndexType,
      Space, typename Space::array_layout>::type;
  ;
  using DenseVector_vil = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, ArrayLayout>, ValueType,
      IndexType, DefaultSpace, ArrayLayout>::type;
  ;
  using DenseVector_vils = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, IndexType, ArrayLayout, Space>,
      ValueType, IndexType, Space, ArrayLayout>::type;
  ;
  using DenseVector_vls = typename Impl::ContainerTester<
      Morpheus::DenseVector<ValueType, ArrayLayout, Space>, ValueType,
      DefaultIndexType, Space, ArrayLayout>::type;
  ;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct DenseMatrixTypes {
  using DefaultValueType   = double;
  using DefaultIndexType   = int;
  using DefaultSpace       = Kokkos::DefaultExecutionSpace;
  using DefaultArrayLayout = typename DefaultSpace::array_layout;

  using DenseMatrix_v =
      typename Impl::ContainerTester<Morpheus::DenseMatrix<ValueType>,
                                     ValueType, DefaultIndexType, DefaultSpace,
                                     DefaultArrayLayout>::type;

  using DenseMatrix_vl = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, ArrayLayout>, ValueType,
      DefaultIndexType, DefaultSpace, ArrayLayout>::type;

  using DenseMatrix_vis = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, Space>, ValueType, IndexType,
      Space, typename Space::array_layout>::type;
  ;
  using DenseMatrix_vil = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout>, ValueType,
      IndexType, DefaultSpace, ArrayLayout>::type;
  ;
  using DenseMatrix_vils = typename Impl::ContainerTester<
      Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout, Space>,
      ValueType, IndexType, Space, ArrayLayout>::type;
  ;
  using DenseMatrix_vls = typename Impl::DenseMatrixTester<
      Morpheus::DenseVector<ValueType, ArrayLayout, Space>, ValueType,
      DefaultIndexType, Space, ArrayLayout>::type;
  ;
};

#endif  // MORPHEUS_CORE_TESTS_TYPEDEFINITION_UTILS_HPP