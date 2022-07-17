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

    typename DenseVector::HostMirror xh_(sz_, 0), yh_(sz_, 0);
    for (int i = 0; i < sz_; i++) {
      xh_(i) = i + 1;
      yh_(i) = sz_ - i;
      result += (i + 1) * (sz_ - i);
    }

    Morpheus::copy(xh_, x_);
    Morpheus::copy(yh_, y_);
  }
};

template <typename T>
inline void MORPHEUS_EXPECT_EQ(T val1, T val2) {
  if (std::is_floating_point<T>::value) {
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<T>, val1,
                        val2);
  } else {
    EXPECT_EQ(val1, val2);
  }
}

namespace Test {

TYPED_TEST_CASE(DotTest, ContainerImplementations);

TYPED_TEST(DotTest, SmallTest) {
  using value_type = typename TestFixture::DenseVector::value_type;
  using index_type = typename TestFixture::DenseVector::index_type;

  index_type sz  = this->s_size;
  value_type res = this->s_res;
  auto x = this->s_x, y = this->s_y;

  auto result = Morpheus::dot<TEST_EXECSPACE>(sz, x, y);

  // Make sure the correct type is returned by dot
  EXPECT_EQ((std::is_same<decltype(result), decltype(res)>::value), 1);

  if (std::is_floating_point<value_type>::value) {
    EXPECT_PRED_FORMAT2(
        ::testing::internal::CmpHelperFloatingPointEQ<value_type>, res, result);
  } else {
    EXPECT_EQ(res, result);
  }
}

// TYPED_TEST(DotTest, MediumTest) {
//   using value_type = typename TestFixture::DenseVector::value_type;
//   using index_type = typename TestFixture::DenseVector::index_type;

//   index_type sz  = this->m_size;
//   value_type res = this->m_res;
//   auto x = this->m_x, y = this->m_y;

//   auto result = Morpheus::dot<TEST_EXECSPACE>(sz, x, y);

//   // Make sure the correct type is returned by dot
//   EXPECT_EQ((std::is_same<decltype(result), decltype(res)>::value), 1);

//   if (std::is_floating_point<value_type>::value) {
//     EXPECT_PRED_FORMAT2(
//         ::testing::internal::CmpHelperFloatingPointEQ<value_type>, res,
//         result);
//   } else {
//     EXPECT_EQ(res, result);
//   }
// }

// TYPED_TEST(DotTest, LargeTest) {
//   using value_type = typename TestFixture::DenseVector::value_type;
//   using index_type = typename TestFixture::DenseVector::index_type;

//   index_type sz  = this->l_size;
//   value_type res = this->l_res;
//   auto x = this->l_x, y = this->l_y;

//   auto result = Morpheus::dot<TEST_EXECSPACE>(sz, x, y);

//   // Make sure the correct type is returned by dot
//   EXPECT_EQ((std::is_same<decltype(result), decltype(res)>::value), 1);

//   if (std::is_floating_point<value_type>::value) {
//     EXPECT_PRED_FORMAT2(
//         ::testing::internal::CmpHelperFloatingPointEQ<value_type>, res,
//         result);
//   } else {
//     EXPECT_EQ(res, result);
//   }
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_DOT_HPP
