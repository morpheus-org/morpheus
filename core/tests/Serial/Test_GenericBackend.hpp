/**
 * Test_GenericBackend.hpp
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

#ifndef TEST_CORE_TEST_GENERICBACKEND_HPP
#define TEST_CORE_TEST_GENERICBACKEND_HPP

#include <Morpheus_Core.hpp>

namespace Test {
/**
 * @brief The \p is_generic_space checks if the passed type is a valid Generic
 * backend. For the check to be valid, the type should be a \p GenericBackend
 * container.
 *
 */
TEST(GenericBackendTest, IsGenericBackendSerial) {
  bool ref_results[7] = {1, 0, 0, 1, 0, 0, 1};

  bool res = Morpheus::is_generic_backend<Kokkos::Serial>::value;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_generic_backend<Morpheus::Serial>::value;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_generic_backend<Morpheus::Custom::Serial>::value;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_generic_backend<Morpheus::Generic::Serial>::value;
  EXPECT_EQ(res, ref_results[3]);
  res = Morpheus::is_generic_backend<typename Morpheus::Serial::backend>::value;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_generic_backend<
      typename Morpheus::Custom::Serial::backend>::value;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_generic_backend<
      typename Morpheus::Generic::Serial::backend>::value;
  EXPECT_EQ(res, ref_results[6]);

  /* Checking Alias */
  res = Morpheus::is_generic_backend_v<Kokkos::Serial>;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_generic_backend_v<Morpheus::Serial>;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_generic_backend_v<Morpheus::Custom::Serial>;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_generic_backend_v<Morpheus::Generic::Serial>;
  EXPECT_EQ(res, ref_results[3]);
  res = Morpheus::is_generic_backend_v<typename Morpheus::Serial::backend>;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_generic_backend_v<
      typename Morpheus::Custom::Serial::backend>;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_generic_backend_v<
      typename Morpheus::Generic::Serial::backend>;
  EXPECT_EQ(res, ref_results[6]);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_GENERICBACKEND_HPP
