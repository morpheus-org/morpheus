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

namespace types {
using value_tlist  = Morpheus::TypeList<double, int>;
using index_tlist  = Morpheus::TypeList<long long, Morpheus::Default>;
using layout_tlist = Morpheus::TypeList<Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                        Morpheus::Default>;
using space_tlist  = Morpheus::TypeList<TEST_EXECSPACE>;
// Generate all unary combinations
using types_set = typename Morpheus::cross_product<
    value_tlist,
    typename Morpheus::cross_product<
        index_tlist, typename Morpheus::cross_product<
                         layout_tlist, space_tlist>::type>::type>::type;

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

template <typename T1, typename T2>
inline void morpheus_expect_eq(T1 val1, T2 val2) {
  if (std::is_floating_point<T1>::value) {
    if ((std::is_floating_point<T2>::value) && (sizeof(T1) > sizeof(T2))) {
      EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<T2>,
                          val1, val2);
    } else if ((std::is_floating_point<T2>::value) &&
               (sizeof(T1) < sizeof(T2))) {
      EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<T1>,
                          val1, val2);
    } else {
      EXPECT_EQ((T2)val1, (T2)val2);
    }
  } else {
    EXPECT_EQ((T1)val1, (T1)val2);
  }
}

#define MORPHEUS_EXPECT_EQ(val1, val2) \
  { morpheus_expect_eq(val1, val2); }

#endif  // TEST_CORE_UTILS_HPP