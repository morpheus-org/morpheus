/**
 * Test_CustomBackend.hpp
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

#ifndef TEST_CORE_HIP_TEST_CUSTOMBACKEND_HPP
#define TEST_CORE_HIP_TEST_CUSTOMBACKEND_HPP

#include <Morpheus_Core.hpp>

namespace Test {

/**
 * @brief The \p is_custom_space checks if the passed type is a valid Custom
 * backend. For the check to be valid, the type should be a \p CustomBackend
 * container.
 *
 */
TEST(CustomBackendTest, IsCustomBackendHIP) {
  bool ref_results[7] = {0, 1, 1, 0, 1, 1, 0};

  bool res = Morpheus::is_custom_backend<Kokkos::HIP>::value;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_custom_backend<Morpheus::HIP>::value;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_custom_backend<Morpheus::Custom::HIP>::value;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_custom_backend<Morpheus::Generic::HIP>::value;
  EXPECT_EQ(res, ref_results[3]);
  res = Morpheus::is_custom_backend<typename Morpheus::HIP::backend>::value;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_custom_backend<
      typename Morpheus::Custom::HIP::backend>::value;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_custom_backend<
      typename Morpheus::Generic::HIP::backend>::value;
  EXPECT_EQ(res, ref_results[6]);

  /* Checking Alias */
  res = Morpheus::is_custom_backend_v<Kokkos::HIP>;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_custom_backend_v<Morpheus::HIP>;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_custom_backend_v<Morpheus::Custom::HIP>;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_custom_backend_v<Morpheus::Generic::HIP>;
  EXPECT_EQ(res, ref_results[3]);
  res = Morpheus::is_custom_backend_v<typename Morpheus::HIP::backend>;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_custom_backend_v<typename Morpheus::Custom::HIP::backend>;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_custom_backend_v<typename Morpheus::Generic::HIP::backend>;
  EXPECT_EQ(res, ref_results[6]);
}

TEST(CustomBackendTest, IsCustomBackendHIPSpace) {
  bool ref_resultsults[7] = {0, 1, 1, 0, 1, 1, 0};

  bool res = Morpheus::is_custom_backend<Kokkos::HIPSpace>::value;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_custom_backend<Morpheus::HIPSpace>::value;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_custom_backend<Morpheus::Custom::HIPSpace>::value;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_custom_backend<Morpheus::Generic::HIPSpace>::value;
  EXPECT_EQ(res, ref_results[3]);
  res =
      Morpheus::is_custom_backend<typename Morpheus::HIPSpace::backend>::value;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_custom_backend<
      typename Morpheus::Custom::HIPSpace::backend>::value;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_custom_backend<
      typename Morpheus::Generic::HIPSpace::backend>::value;
  EXPECT_EQ(res, ref_results[6]);

  /* Checking Alias */
  res = Morpheus::is_custom_backend_v<Kokkos::HIPSpace>;
  EXPECT_EQ(res, ref_results[0]);
  res = Morpheus::is_custom_backend_v<Morpheus::HIPSpace>;
  EXPECT_EQ(res, ref_results[1]);
  res = Morpheus::is_custom_backend_v<Morpheus::Custom::HIPSpace>;
  EXPECT_EQ(res, ref_results[2]);
  res = Morpheus::is_custom_backend_v<Morpheus::Generic::HIPSpace>;
  EXPECT_EQ(res, ref_results[3]);
  res = Morpheus::is_custom_backend_v<typename Morpheus::HIPSpace::backend>;
  EXPECT_EQ(res, ref_results[4]);
  res = Morpheus::is_custom_backend_v<
      typename Morpheus::Custom::HIPSpace::backend>;
  EXPECT_EQ(res, ref_results[5]);
  res = Morpheus::is_custom_backend_v<
      typename Morpheus::Generic::HIPSpace::backend>;
  EXPECT_EQ(res, ref_results[6]);
}

}  // namespace Test

#endif  // TEST_CORE_HIP_TEST_CUSTOMBACKEND_HPP
