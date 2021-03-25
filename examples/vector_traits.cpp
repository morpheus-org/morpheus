/**
 * vector_traits.cpp
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

#include <morpheus/core/vector_traits.hpp>

// Template argument options:
// - VectorTraits<ValueType>
// - VectorTraits<ValueType, Space>

using traits_d = Morpheus::Impl::VectorTraits<double>;
static_assert(std::is_same_v<traits_d::value_type, double>);

using traits_i = Morpheus::Impl::VectorTraits<int>;
static_assert(std::is_same_v<traits_i::value_type, int>);

using traits_l = Morpheus::Impl::VectorTraits<long>;
static_assert(std::is_same_v<traits_l::value_type, long>);

using traits_f = Morpheus::Impl::VectorTraits<float>;
static_assert(std::is_same_v<traits_f::value_type, float>);

using traits_fv = Morpheus::Impl::VectorTraits<float, void>;
static_assert(std::is_same_v<traits_fv::value_type, float>);
static_assert(
    std::is_same_v<traits_fv::memory_space, Kokkos::HostSpace::memory_space>);
static_assert(
    std::is_same_v<traits_fv::execution_space, Kokkos::DefaultExecutionSpace>);

using traits_fh = Morpheus::Impl::VectorTraits<float, Kokkos::HostSpace>;
static_assert(std::is_same_v<traits_fh::value_type, float>);
static_assert(
    std::is_same_v<traits_fh::memory_space, Kokkos::HostSpace::memory_space>);
static_assert(std::is_same_v<traits_fh::execution_space,
                             Kokkos::HostSpace::execution_space>);

int main() { std::puts("All checks are passed."); }