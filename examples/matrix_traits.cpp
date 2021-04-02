/**
 * matrix_traits.cpp
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

#include <iostream>

#include <morpheus/core/matrix_traits.hpp>

//  * Template argument options:
//  *  - MatrixTraits<ValueType>
//  *  - MatrixTraits<ValueType, ArrayLayout>
//  *  - MatrixTraits<ValueType, IndexType, Space>
//  *  - MatrixTraits<ValueType, IndexType, ArrayLayout>
//  *  - MatrixTraits<ValueType, IndexType, ArrayLayout, Space>
//  *  - MatrixTraits<ValueType, ArrayLayout, Space>

//   using value_type   = ValueType;
//   using index_type   = IndexType;
//   using array_layout = ArrayLayout;

//   using execution_space = ExecutionSpace;
//   using memory_space    = MemorySpace;
//   using device_type     = Kokkos::Device<ExecutionSpace, MemorySpace>;

// MatrixTraits<ValueType>
using case1 = Morpheus::Impl::MatrixTraits<double>;
static_assert(std::is_same_v<case1::value_type, double>);
static_assert(std::is_same_v<case1::index_type, int>);
static_assert(std::is_same_v<case1::array_layout, Kokkos::LayoutRight>);
static_assert(
    std::is_same_v<case1::execution_space, Kokkos::DefaultExecutionSpace>);
static_assert(std::is_same_v<case1::memory_space,
                             Kokkos::DefaultExecutionSpace::memory_space>);
static_assert(
    std::is_same_v<case1::device_type, Kokkos::Device<case1::execution_space,
                                                      case1::memory_space>>);

// MatrixTraits<ValueType, ArrayLayout>
using case2 = Morpheus::Impl::MatrixTraits<double, Kokkos::LayoutStride>;
static_assert(std::is_same_v<case2::value_type, double>);
static_assert(std::is_same_v<case2::index_type, int>);
static_assert(std::is_same_v<case2::array_layout, Kokkos::LayoutStride>);
static_assert(
    std::is_same_v<case2::execution_space, Kokkos::DefaultExecutionSpace>);
static_assert(std::is_same_v<case2::memory_space,
                             Kokkos::DefaultExecutionSpace::memory_space>);
static_assert(
    std::is_same_v<case2::device_type, Kokkos::Device<case2::execution_space,
                                                      case2::memory_space>>);

// MatrixTraits<ValueType, IndexType, Space>
using case3 = Morpheus::Impl::MatrixTraits<double, long, Kokkos::Serial>;
static_assert(std::is_same_v<case3::value_type, double>);
static_assert(std::is_same_v<case3::index_type, long>);
static_assert(std::is_same_v<case3::array_layout, Kokkos::LayoutRight>);
static_assert(
    std::is_same_v<case3::execution_space, Kokkos::Serial::execution_space>);
static_assert(
    std::is_same_v<case3::memory_space, Kokkos::Serial::memory_space>);
static_assert(
    std::is_same_v<case3::device_type, Kokkos::Device<case3::execution_space,
                                                      case3::memory_space>>);

// MatrixTraits<ValueType, IndexType, ArrayLayout>
using case4 = Morpheus::Impl::MatrixTraits<double, long, Kokkos::LayoutStride>;
static_assert(std::is_same_v<case4::value_type, double>);
static_assert(std::is_same_v<case4::index_type, long>);
static_assert(std::is_same_v<case4::array_layout, Kokkos::LayoutStride>);
static_assert(std::is_same_v<case4::execution_space,
                             Kokkos::DefaultExecutionSpace::execution_space>);
static_assert(std::is_same_v<case4::memory_space,
                             Kokkos::DefaultExecutionSpace::memory_space>);
static_assert(
    std::is_same_v<case4::device_type, Kokkos::Device<case4::execution_space,
                                                      case4::memory_space>>);

// MatrixTraits<ValueType, IndexType, ArrayLayout, Space>
using case5 = Morpheus::Impl::MatrixTraits<double, long, Kokkos::LayoutStride,
                                           Kokkos::Serial>;
static_assert(std::is_same_v<case5::value_type, double>);
static_assert(std::is_same_v<case5::index_type, long>);
static_assert(std::is_same_v<case5::array_layout, Kokkos::LayoutStride>);
static_assert(
    std::is_same_v<case5::execution_space, Kokkos::Serial::execution_space>);
static_assert(
    std::is_same_v<case5::memory_space, Kokkos::Serial::memory_space>);
static_assert(
    std::is_same_v<case5::device_type, Kokkos::Device<case5::execution_space,
                                                      case5::memory_space>>);

// MatrixTraits<ValueType, ArrayLayout, Space>
using case6 =
    Morpheus::Impl::MatrixTraits<double, Kokkos::LayoutStride, Kokkos::Serial>;
static_assert(std::is_same_v<case6::value_type, double>);
static_assert(std::is_same_v<case6::index_type, int>);
static_assert(std::is_same_v<case6::array_layout, Kokkos::LayoutStride>);
static_assert(
    std::is_same_v<case6::execution_space, Kokkos::Serial::execution_space>);
static_assert(
    std::is_same_v<case6::memory_space, Kokkos::Serial::memory_space>);
static_assert(
    std::is_same_v<case6::device_type, Kokkos::Device<case6::execution_space,
                                                      case6::memory_space>>);

int main() { std::puts("All checks are passed."); }