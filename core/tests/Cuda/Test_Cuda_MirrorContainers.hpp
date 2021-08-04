/**
 * Test_Cuda_MirrorContainers.hpp
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
TEST(TESTSUITE_NAME, Cuda_MirrorContainer_DenseVector_new_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror_container(x);

  using mirror = decltype(x_mirror);
  ::testing::StaticAssertTypeEq<typename mirror::type,
                                typename vector::HostMirror>();
  ASSERT_EQ(x.size(), x_mirror.size());

  for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror should be the default "
                                 "(0) i.e no copy was performed";
  }
}

}  // namespace Test