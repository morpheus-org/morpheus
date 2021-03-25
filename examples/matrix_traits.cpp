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

// Template argument options:
// - MatrixTraits<ValueType>
// - MatrixTraits<ValueType, IndexType>
//- MatrixTraits<ValueType, IndexType, Space>
//- MatrixTraits<ValueType, Space>

using traits_d = Morpheus::Impl::MatrixTraits<double>;
static_assert(std::is_same_v<traits_d::value_type, double>);
static_assert(std::is_same_v<traits_d::index_type, int>);

using traits_il = Morpheus::Impl::MatrixTraits<int, long>;
static_assert(std::is_same_v<traits_il::value_type, int>);
static_assert(std::is_same_v<traits_il::index_type, long>);

using traits_l = Morpheus::Impl::MatrixTraits<long>;
static_assert(std::is_same_v<traits_l::value_type, long>);
static_assert(std::is_same_v<traits_l::index_type, int>);

using traits_flv = Morpheus::Impl::MatrixTraits<float, long>;
static_assert(std::is_same_v<traits_flv::value_type, float>);
static_assert(std::is_same_v<traits_flv::index_type, long>);
static_assert(
    std::is_same_v<traits_flv::memory_space, Kokkos::HostSpace::memory_space>);
static_assert(
    std::is_same_v<traits_flv::execution_space, Kokkos::DefaultExecutionSpace>);

using traits_sp = Morpheus::Impl::MatrixTraits<float, long, Kokkos::HostSpace>;
static_assert(std::is_same_v<traits_sp::value_type, float>);
static_assert(std::is_same_v<traits_sp::index_type, long>);
static_assert(
    std::is_same_v<traits_sp::memory_space, Kokkos::HostSpace::memory_space>);
static_assert(std::is_same_v<traits_sp::execution_space,
                             Kokkos::HostSpace::execution_space>);

int main() { std::puts("All checks are passed."); }