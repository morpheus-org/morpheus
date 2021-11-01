/**
 * Test_Dot.hpp
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

#ifndef TEST_CORE_TEST_DOT_HPP
#define TEST_CORE_TEST_DOT_HPP

#include <Morpheus_Core.hpp>

#include <setup/ContainerDefinition_Utils.hpp>

template <typename ContainerImplementations>
class DotTest : public ::testing::Test {
 public:
  using DenseVector = typename ContainerImplementations::DenseVector;

  // You can define per-test set-up logic as usual.
  void SetUp() override {
    local_setup(s_x, s_y, s_size, s_res);
    local_setup(m_x, m_y, m_size, m_res);
    local_setup(l_x, l_y, l_size, l_res);
  }

  // You can define per-test tear-down logic as usual.
  void TearDown() override {}

  DenseVector s_x, s_y;
  DenseVector m_x, m_y;
  DenseVector l_x, l_y;
  typename DenseVector::index_type s_size = 1, m_size = 100, l_size = 1000;
  typename DenseVector::value_type s_res = 0, m_res = 0, l_res = 0;

 private:
  void local_setup(DenseVector& x_, DenseVector& y_,
                   typename DenseVector::index_type sz_,
                   typename DenseVector::value_type& result) {
    x_.resize(sz_);
    y_.resize(sz_);

    for (int i = 0; i < sz_; i++) {
      x_(i) = i;
      y_(i) = sz_ - i;
      result += i * (sz_ - i);
    }
  }
};

namespace Test {

TYPED_TEST_CASE(DotTest, ContainerImplementations);

TYPED_TEST(DotTest, SmallTest) {
  typename TestFixture::DenseVector::index_type sz  = this->s_size;
  typename TestFixture::DenseVector::value_type res = this->s_res;

  auto result = Morpheus::dot<TEST_EXECSPACE>(sz, this->s_x, this->s_y);

  std::cout << "SMALL_RESULT = " << result << std::endl;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, result, res);
}

TYPED_TEST(DotTest, MediumTest) {
  typename TestFixture::DenseVector::index_type sz  = this->m_size;
  typename TestFixture::DenseVector::value_type res = this->m_res;

  auto result = Morpheus::dot<TEST_EXECSPACE>(sz, this->m_x, this->m_y);

  std::cout << "MEDIUM_RESULT = " << result << std::endl;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, result, res);
}

TYPED_TEST(DotTest, LargeTest) {
  typename TestFixture::DenseVector::index_type sz  = this->l_size;
  typename TestFixture::DenseVector::value_type res = this->l_res;

  auto result = Morpheus::dot<TEST_EXECSPACE>(sz, this->l_x, this->l_y);

  std::cout << "LARGE_RESULT = " << result << std::endl;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, result, res);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DOT_HPP
