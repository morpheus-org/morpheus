/**
 * Test_GenericSpace.hpp
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

#ifndef TEST_CORE_TEST_GENERICSPACE_HPP
#define TEST_CORE_TEST_GENERICSPACE_HPP

#include <Morpheus_Core.hpp>

namespace Test {
/**
 * @brief The \p is_generic_space checks if the passed type is a valid Generic
 * execution space. For the check to be valid, the type should be a
 * \p GenericSpace container.
 *
 */
TEST(GenericSpaceTest, IsGenericSpace) {
  bool res = Morpheus::is_generic_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_generic_space<Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_generic_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_generic_space<Morpheus::GenericSpace<Kokkos::Serial>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_generic_space<Morpheus::Generic::Serial>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_generic_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_generic_space<Morpheus::Generic::OpenMP>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_generic_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_generic_space<Morpheus::Generic::Cuda>::value;
  EXPECT_EQ(res, 1);
#endif

  /* Testing Alias */
  res = Morpheus::is_generic_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_generic_space_v<Morpheus::GenericSpace<Kokkos::Serial>>;
  EXPECT_EQ(res, 1);
#endif
}

/**
 * @brief The \p has_generic_space checks if the passed type has a valid
 * Generic execution space. For the check to be valid, the type should have a
 * \p generic_space trait which is a valid generic space.
 *
 */
TEST(GenericSpaceTest, HasGenericSpace) {
  bool res = Morpheus::has_generic_space<int>::value;
  EXPECT_EQ(res, 0);

  struct TestNotGeneric {
    using generic_space = Kokkos::DefaultHostExecutionSpace;
  };

  // TestNotGeneric has a generic_space trait but that is not a generic space
  res = Morpheus::has_generic_space<TestNotGeneric>::value;
  EXPECT_EQ(res, 0);

  struct TestGeneric {
    using generic_space =
        Morpheus::GenericSpace<Kokkos::DefaultHostExecutionSpace>;
  };

  // TestNotGeneric has a generic_space trait which is also a generic space
  res = Morpheus::has_generic_space<TestGeneric>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  // TestNotGeneric has a generic_space trait but that is not a generic space
  res = Morpheus::has_generic_space_v<TestNotGeneric>;
  EXPECT_EQ(res, 0);

  // TestNotGeneric has a generic_space trait which is also a generic space
  res = Morpheus::has_generic_space_v<TestGeneric>;
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_GENERICSPACE_HPP
