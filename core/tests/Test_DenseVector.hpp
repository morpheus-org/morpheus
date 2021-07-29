/**
 * Test_DenseVector.hpp
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

#include <Morpheus_Core.hpp>

namespace Test {
TEST(TESTSUITE_NAME, densevectortraits) {
  using vec = Morpheus::DenseVector<double, long long, Kokkos::LayoutRight,
                                    TEST_EXECSPACE>;
  ::testing::StaticAssertTypeEq<typename vec::value_type, double>();
  ::testing::StaticAssertTypeEq<typename vec::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vec::array_layout,
                                Kokkos::LayoutRight>();
  ::testing::StaticAssertTypeEq<typename vec::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vec::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<typename vec::device_type,
                                typename TEST_EXECSPACE::device_type>();
  ::testing::StaticAssertTypeEq<
      typename vec::host_mirror_type,
      Morpheus::DenseVector<typename vec::non_const_value_type,
                            typename vec::array_layout,
                            typename vec::traits::host_mirror_space>>();
}
}  // namespace Test