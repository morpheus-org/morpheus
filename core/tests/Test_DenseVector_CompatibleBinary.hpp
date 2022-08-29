/**
 * Test_DenseVector_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_DENSEVECTOR_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_DENSEVECTOR_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

using DenseVectorCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DenseVector<double>, types::compatible_types_set>::type;

using CompatibleDenseVectorBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DenseVectorCompatibleTypes, DenseVectorCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleDenseVectorBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // DenseVector
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // DenseVector
  using host2   = typename type2::type::HostMirror;
};

namespace Test {
/**
 * @brief Test Suite using the Compatible Binary DenseVectors
 *
 */
TYPED_TEST_SUITE(CompatibleDenseVectorBinaryTest, CompatibleDenseVectorBinary);

/**
 * @brief Testing shallow copy construction of DenseVector container from
 * another DenseVector container with the different parameters. Resulting
 * container should be a shallow copy of the original.
 *
 */
TYPED_TEST(CompatibleDenseVectorBinaryTest, ShallowCopyConstructor) {
  using Vector1     = typename TestFixture::device1;
  using HostVector1 = typename TestFixture::host1;
  using Vector2     = typename TestFixture::device2;
  using HostVector2 = typename TestFixture::host2;
  using value_type  = typename Vector1::value_type;

  auto size = 10;
  Vector1 x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector1 xh(x.size(), (value_type)0);
  Morpheus::copy(x, xh);

  HostVector2 yh(xh);
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  for (size_t i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector2 y(x);
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector2 yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_t i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

/**
 * @brief Testing shallow copy assignment of DenseVector container from
 * another DenseVector container with the different parameters. Resulting
 * copy should be a shallow copy from the original.
 *
 */
TYPED_TEST(CompatibleDenseVectorBinaryTest, ShallowCopyAssignment) {
  using Vector1     = typename TestFixture::device1;
  using HostVector1 = typename TestFixture::host1;
  using Vector2     = typename TestFixture::device2;
  using HostVector2 = typename TestFixture::host2;
  using value_type  = typename Vector1::value_type;

  auto size = 10;
  Vector1 x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector1 xh(x.size(), 0);
  Morpheus::copy(x, xh);

  HostVector2 yh = xh;
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  for (size_t i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector2 y = x;
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector2 yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_t i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_COMPATIBLEBINARY_HPP
