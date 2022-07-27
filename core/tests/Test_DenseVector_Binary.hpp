/**
 * Test_DenseVector_Binary.hpp
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

#ifndef TEST_CORE_TEST_DENSEVECTOR_BINARY_HPP
#define TEST_CORE_TEST_DENSEVECTOR_BINARY_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;

using DenseVectorBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DenseVectorTypes, DenseVectorTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class DenseVectorBinaryTest : public ::testing::Test {
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
TYPED_TEST_CASE(DenseVectorBinaryTest, DenseVectorBinary);

/**
 * @brief Testing allocation of DenseVector container from another DenseVector
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(DenseVectorBinaryTest, Allocate) {
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

  HostVector2 yh;
  EXPECT_EQ(yh.size(), 0);
  EXPECT_EQ(yh.view().size(), 0);

  yh.allocate(xh);
  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  EXPECT_EQ(yh.size(), xh.size());
  EXPECT_EQ(yh.view().size(), xh.view().size());
  for (size_t i = 0; i < xh.size(); i++) {
    EXPECT_EQ(yh[i], (value_type)0);
  }

  // Now check device vector
  Vector2 y;
  EXPECT_EQ(y.size(), 0);
  EXPECT_EQ(y.view().size(), 0);

  yh[4] = (value_type)-4.33;
  yh[9] = (value_type)-9.44;
  y.allocate(x);
  Morpheus::copy(y, yh);

  EXPECT_EQ(y.size(), yh.size());
  EXPECT_EQ(y.view().size(), yh.view().size());
  for (size_t i = 0; i < yh.size(); i++) {
    EXPECT_EQ(yh[i], (value_type)0);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_HPP
