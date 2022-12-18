/**
 * Test_Spaces.hpp
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

#ifndef TEST_CORE_SERIAL_TEST_SPACES_HPP
#define TEST_CORE_SERIAL_TEST_SPACES_HPP

#include <Morpheus_Core.hpp>

namespace Test {

TEST(SpacesTest, HostMirrorSerial) {
  if (std::is_same<Morpheus::Serial,
                   Morpheus::DefaultHostExecutionSpace>::value) {
    using exe    = Morpheus::Serial;
    using mirror = Morpheus::HostMirror<exe>;
    // keep_exe & keep_mem
    bool res = std::is_same<typename mirror::backend, exe>::value;
    EXPECT_EQ(res, 1);
  }
}

TEST(SpacesTest, HasBackendSerial) {
  bool backend_results[4] = {0, 1, 1, 1};

  bool res = Morpheus::has_backend<Kokkos::Serial>::value;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend<Morpheus::Serial>::value;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend<Morpheus::Custom::Serial>::value;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend<Morpheus::Generic::Serial>::value;
  EXPECT_EQ(res, backend_results[3]);
  /* Checking Alias */
  res = Morpheus::has_backend_v<Kokkos::Serial>;
  EXPECT_EQ(res, backend_results[0]);
  res = Morpheus::has_backend_v<Morpheus::Serial>;
  EXPECT_EQ(res, backend_results[1]);
  res = Morpheus::has_backend_v<Morpheus::Custom::Serial>;
  EXPECT_EQ(res, backend_results[2]);
  res = Morpheus::has_backend_v<Morpheus::Generic::Serial>;
  EXPECT_EQ(res, backend_results[3]);
}

}  // namespace Test

#endif  // TEST_CORE_SERIAL_TEST_SPACES_HPP
