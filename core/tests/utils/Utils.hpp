/**
 * Utils.hpp
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

#ifndef TEST_CORE_UTILS_HPP
#define TEST_CORE_UTILS_HPP

#include <gtest/gtest.h>
#include <Morpheus_Core.hpp>

namespace Morpheus {
MORPHEUS_IMPL_HAS_TRAIT(type)
MORPHEUS_IMPL_HAS_TRAIT(traits)
}  // namespace Morpheus

namespace types {
using value_tlist  = Morpheus::TypeList<double, int>;
using index_tlist  = Morpheus::TypeList<long long, Morpheus::Default>;
using layout_tlist = Morpheus::TypeList<Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                        Morpheus::Default>;
// using space_tlist  = Morpheus::TypeList<TEST_EXECSPACE>;
using space_tlist = Morpheus::TypeList<TEST_CUSTOM_SPACE>;
// Generate all unary combinations
using types_set = typename Morpheus::cross_product<
    value_tlist,
    typename Morpheus::cross_product<
        index_tlist, typename Morpheus::cross_product<
                         layout_tlist, space_tlist>::type>::type>::type;

using convert_value_tlist = Morpheus::TypeList<double, int>;
using convert_index_tlist = Morpheus::TypeList<long long>;
using convert_layout_tlist =
    Morpheus::TypeList<Kokkos::LayoutRight, Kokkos::LayoutLeft>;
using convert_space_tlist = Morpheus::TypeList<TEST_CUSTOM_SPACE>;
// Generate all unary combinations
using convert_types_set = typename Morpheus::cross_product<
    convert_value_tlist,
    typename Morpheus::cross_product<
        convert_index_tlist,
        typename Morpheus::cross_product<
            convert_layout_tlist, convert_space_tlist>::type>::type>::type;

// Generate compatible unary combinations
using compatible_value_tlist = Morpheus::TypeList<double>;
using compatible_index_tlist = Morpheus::TypeList<int, Morpheus::Default>;
using compatible_layout_tlist =
    Morpheus::TypeList<typename TEST_EXECSPACE::array_layout,
                       Morpheus::Default>;

// Generate all compatible unary combinations
using compatible_types_set = typename Morpheus::cross_product<
    compatible_value_tlist,
    typename Morpheus::cross_product<
        compatible_index_tlist,
        typename Morpheus::cross_product<compatible_layout_tlist,
                                         space_tlist>::type>::type>::type;

/**
 * Single type list for rapid testing
 *
 */
using test_value_tlist = Morpheus::TypeList<double>;
using test_index_tlist = Morpheus::TypeList<int>;
using test_layout_tlist =
    Morpheus::TypeList<typename TEST_EXECSPACE::array_layout>;

// Generate all compatible unary combinations
using test_types_set = typename Morpheus::cross_product<
    test_value_tlist,
    typename Morpheus::cross_product<
        test_index_tlist,
        typename Morpheus::cross_product<test_layout_tlist,
                                         space_tlist>::type>::type>::type;

}  // namespace types

template <typename... Ts>
struct to_gtest_types {};

// Convert a Morpheus TypeList to Gtest Types
template <typename... Ts>
struct to_gtest_types<Morpheus::TypeList<Ts...>> {
  using type = ::testing::Types<Ts...>;
};

// TODO: Move in Morpheus Core
template <typename... T>
struct generate_pair;
// Partially specialise the empty cases.
template <typename... Us>
struct generate_pair<Morpheus::TypeList<>, Morpheus::TypeList<Us...>> {
  using type = Morpheus::TypeList<>;
};

template <typename... Us>
struct generate_pair<Morpheus::TypeList<Us...>, Morpheus::TypeList<>> {
  using type = Morpheus::TypeList<>;
};

template <>
struct generate_pair<Morpheus::TypeList<>, Morpheus::TypeList<>> {
  using type = Morpheus::TypeList<>;
};

template <typename T, typename... Ts, typename U, typename... Us>
struct generate_pair<Morpheus::TypeList<T, Ts...>,
                     Morpheus::TypeList<U, Us...>> {
  using type = typename Morpheus::concat<
      Morpheus::TypeList<std::pair<T, U>>,
      typename generate_pair<Morpheus::TypeList<Ts...>,
                             Morpheus::TypeList<Us...>>::type>::type;
};

#endif  // TEST_CORE_UTILS_HPP