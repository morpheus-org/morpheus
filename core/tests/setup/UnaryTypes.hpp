/**
 * UnaryTypes.hpp
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

#ifndef MORPHEUS_CORE_TESTS_SETUP_UNARYTYPES_HPP
#define MORPHEUS_CORE_TESTS_SETUP_UNARYTYPES_HPP

#include <gtest/gtest.h>
#include <Morpheus_Core.hpp>
#include <TestUtils.hpp>

namespace unary {
using value_tlist  = Morpheus::TypeList<double, float, int>;
using index_tlist  = Morpheus::TypeList<int, long long, Morpheus::Default>;
using layout_tlist = Morpheus::TypeList<Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                        Morpheus::Default>;
using space_tlist  = Morpheus::TypeList<TEST_EXECSPACE>;
// Generate all unary combinations
using types_set = typename Morpheus::cross_product<
    value_tlist,
    typename Morpheus::cross_product<
        index_tlist, typename Morpheus::cross_product<
                         layout_tlist, space_tlist>::type>::type>::type;

using DenseVector =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types_set>::type;

using DenseMatrix =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types_set>::type;

using CooMatrix =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types_set>::type;

using CsrMatrix =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types_set>::type;

using DiaMatrix =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types_set>::type;

using DynamicMatrix =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types_set>::type;
}  // namespace unary

#endif  // MORPHEUS_CORE_TESTS_SETUP_UNARYTYPES_HPP
