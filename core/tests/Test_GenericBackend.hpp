/**
 * Test_GenericBackend.hpp
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

#ifndef TEST_CORE_TEST_GENERICBACKEND_HPP
#define TEST_CORE_TEST_GENERICBACKEND_HPP

#include <Morpheus_Core.hpp>

namespace Test {
#define MORPHEUS_CHECK_GENERIC_BACKEND(SPACE, ref_res)                        \
  {                                                                           \
    bool _res;                                                                \
    _res = Morpheus::is_generic_backend<Kokkos::SPACE>::value;                \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::SPACE>::value;              \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::Custom::SPACE>::value;      \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::Generic::SPACE>::value;     \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::SPACE::backend>::value;                            \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::Custom::SPACE::backend>::value;                    \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::Generic::SPACE::backend>::value;                   \
    EXPECT_EQ(_res, ref_res[6]);                                              \
    /* Checking Alias */                                                      \
    _res = Morpheus::is_generic_backend_v<Kokkos::SPACE>;                     \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::SPACE>;                   \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::Custom::SPACE>;           \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::Generic::SPACE>;          \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::is_generic_backend_v<typename Morpheus::SPACE::backend>; \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::is_generic_backend_v<                                    \
        typename Morpheus::Custom::SPACE::backend>;                           \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::is_generic_backend_v<                                    \
        typename Morpheus::Generic::SPACE::backend>;                          \
    EXPECT_EQ(_res, ref_res[6]);                                              \
  }

/**
 * @brief The \p is_generic_space checks if the passed type is a valid Generic
 * backend. For the check to be valid, the type should be a \p GenericBackend
 * container.
 *
 */
TEST(GenericBackendTest, IsGenericBackend) {
  {
    bool res = Morpheus::is_generic_backend<int>::value;
    EXPECT_EQ(res, 0);
  }

  {
    bool ref_results[7] = {1, 0, 0, 1, 0, 0, 1};
    MORPHEUS_CHECK_GENERIC_BACKEND(DefaultHostExecutionSpace, ref_results);
  }

  {
    bool ref_results[7] = {1, 0, 0, 1, 0, 0, 1};
    MORPHEUS_CHECK_GENERIC_BACKEND(TEST_SPACE, ref_results);
  }

  {
    bool ref_results[7] = {1, 0, 0, 1, 0, 0, 1};
    MORPHEUS_CHECK_GENERIC_BACKEND(HostSpace, ref_results);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_GENERICBACKEND_HPP
