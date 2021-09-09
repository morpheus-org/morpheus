/**
 * TestDenseVector_Utils.hpp
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

#ifndef MORPHEUS_CORE_TEST_DENSEVECTOR_UTILS_HPP
#define MORPHEUS_CORE_TEST_DENSEVECTOR_UTILS_HPP

#include <Morpheus_Core.hpp>

template <typename Container>
void test_traits(Morpheus::DenseVectorTag) {
  static_assert(std::is_same<typename Container::value_array_pointer,
                             typename std::add_pointer<
                                 typename Container::value_type>::type>::value);
  static_assert(std::is_same<typename Container::value_array_reference,
                             typename std::add_lvalue_reference<
                                 typename Container::value_type>::type>::value);
}

template <typename T1, typename T2>
void check_shapes(const T1& x, const T2& x_mirror, Morpheus::DenseVectorTag) {
  ASSERT_EQ(x.size(), x_mirror.size());
  ASSERT_EQ(x_mirror.const_view().extent(0), x.size());
}

#endif  // MORPHEUS_CORE_TEST_DENSEVECTOR_UTILS_HPP