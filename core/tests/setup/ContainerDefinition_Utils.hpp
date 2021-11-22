/**
 * ContainerDefinition_Utils.hpp
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

#ifndef MORPHEUS_CORE_TESTS_CONTAINER_DEFINITION_UTILS_HPP
#define MORPHEUS_CORE_TESTS_CONTAINER_DEFINITION_UTILS_HPP

#include <setup/DenseVectorDefinition_Utils.hpp>
#include <setup/DenseMatrixDefinition_Utils.hpp>
#include <setup/CooMatrixDefinition_Utils.hpp>
#include <setup/CsrMatrixDefinition_Utils.hpp>
#include <setup/DiaMatrixDefinition_Utils.hpp>
#include <setup/DynamicMatrixDefinition_Utils.hpp>

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_v {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::v;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::v;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::v;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::v;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::v;

  using DynamicMatrix =
      typename DynamicMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::v;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_vl {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::vl;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vl;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vl;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vl;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vl;

  using DynamicMatrix =
      typename DynamicMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vl;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_vis {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::vis;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vis;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vis;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vis;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vis;

  using DynamicMatrix = typename DynamicMatrixTypes<ValueType, IndexType,
                                                    ArrayLayout, Space>::vis;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_vil {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::vil;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vil;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vil;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vil;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vil;

  using DynamicMatrix = typename DynamicMatrixTypes<ValueType, IndexType,
                                                    ArrayLayout, Space>::vil;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_vils {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::vils;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vils;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vils;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vils;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vils;

  using DynamicMatrix = typename DynamicMatrixTypes<ValueType, IndexType,
                                                    ArrayLayout, Space>::vils;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
struct ContainerTypes_vls {
  using DenseVector =
      typename DenseVectorTypes<ValueType, IndexType, ArrayLayout, Space>::vls;

  using DenseMatrix =
      typename DenseMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vls;

  using CooMatrix =
      typename CooMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vls;

  using CsrMatrix =
      typename CsrMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vls;

  using DiaMatrix =
      typename DiaMatrixTypes<ValueType, IndexType, ArrayLayout, Space>::vls;

  using DynamicMatrix = typename DynamicMatrixTypes<ValueType, IndexType,
                                                    ArrayLayout, Space>::vls;
};

using ContainerImplementations = ::testing::Types<
    ContainerTypes_v<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    ContainerTypes_vl<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    ContainerTypes_vis<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    ContainerTypes_vil<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    ContainerTypes_vils<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
    ContainerTypes_vls<double, int, Kokkos::LayoutRight, Kokkos::Serial>>;

// TODO: Create similar class for when using different algorithms
// and setup a small, medium and large case vectors with assigned values
// do that in a separate setup file such that it is visible by different
// algorithms

template <typename ContainerImplementations>
class ContainersTest : public ::testing::Test {
 public:
  using DenseVector = typename ContainerImplementations::DenseVector;

  using DenseMatrix = typename ContainerImplementations::DenseMatrix;

  using CooMatrix = typename ContainerImplementations::CooMatrix;

  using CsrMatrix = typename ContainerImplementations::CsrMatrix;

  using DiaMatrix = typename ContainerImplementations::DiaMatrix;

  using DynamicMatrix = typename ContainerImplementations::DynamicMatrix;

  // You can define per-test set-up logic as usual.
  void SetUp() override { x.assign(5, 11); }

  // You can define per-test tear-down logic as usual.
  void TearDown() override {}

  DenseVector x;
};

#endif  // MORPHEUS_CORE_TESTS_CONTAINER_DEFINITION_UTILS_HPP