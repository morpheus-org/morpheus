/**
 * Test_Spaces.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef TEST_CORE_HIP_TEST_SPACES_HPP
#define TEST_CORE_HIP_TEST_SPACES_HPP

#include <Morpheus_Core.hpp>

namespace Test {

TEST(SpacesTest, HostMirrorHIP) {
  using mirror = Morpheus::HostMirror<Morpheus::Cuda>;

  bool res = std::is_same<typename mirror::backend, Morpheus::HostSpace>::value;
  EXPECT_EQ(res, 1);
}

TEST(SpacesTest, HasBackendHIP) {
  bool backend_results[4] = {0, 1, 1, 1};

  bool res = Morpheus::has_backend<Kokkos::HIP>::value;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend<Morpheus::HIP>::value;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend<Morpheus::Custom::HIP>::value;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend<Morpheus::Generic::HIP>::value;
  EXPECT_EQ(res, backend_results[3]);

  /* Checking Alias */
  res = Morpheus::has_backend_v<Kokkos::HIP>;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend_v<Morpheus::HIP>;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend_v<Morpheus::Custom::HIP>;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend_v<Morpheus::Generic::HIP>;
  EXPECT_EQ(res, backend_results[3]);
}

TEST(SpacesTest, HasBackendHIPSpace) {
  bool backend_results[4] = {0, 1, 1, 1};

  bool res = Morpheus::has_backend<Kokkos::HIPSpace>::value;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend<Morpheus::HIPSpace>::value;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend<Morpheus::Custom::HIPSpace>::value;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend<Morpheus::Generic::HIPSpace>::value;
  EXPECT_EQ(res, backend_results[3]);

  /* Checking Alias */
  res = Morpheus::has_backend_v<Kokkos::HIPSpace>;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend_v<Morpheus::HIPSpace>;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend_v<Morpheus::Custom::HIPSpace>;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend_v<Morpheus::Generic::HIPSpace>;
  EXPECT_EQ(res, backend_results[3]);
}

}  // namespace Test

#endif  // TEST_CORE_HIP_TEST_SPACES_HPP
