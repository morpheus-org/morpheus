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
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using DenseVectorPairs =
    generate_pair<DenseVectorTypes, DenseVectorTypes>::type;

using DotTypes = to_gtest_types<DenseVectorPairs>::type;

template <typename Containers>
class DotTypesTest : public ::testing::Test {
 public:
  using type       = Containers;
  using src_t      = typename Containers::first_type::type;
  using dst_t      = typename Containers::second_type::type;
  using src_dev_t  = typename src_t::type;
  using src_host_t = typename src_t::type::HostMirror;
  using dst_dev_t  = typename dst_t::type;
  using dst_host_t = typename dst_t::type::HostMirror;
  using IndexType  = typename src_dev_t::index_type;
  using ValueType  = typename dst_dev_t::value_type;

  src_dev_t small_x, med_x, large_x;
  dst_dev_t small_y, med_y, large_y;

  IndexType small_size = 50, med_size = 5000, large_size = 50000;
  ValueType small_ref_res = 0, med_ref_res = 0, large_ref_res = 0;

  // You can define per-test set-up logic as usual.
  void SetUp() override {
    local_setup(small_x, small_y, small_size, small_ref_res);
    local_setup(med_x, med_y, med_size, med_ref_res);
    local_setup(large_x, large_y, large_size, large_ref_res);
  }

 private:
  void local_setup(src_dev_t& x_, dst_dev_t& y_, IndexType sz_,
                   ValueType& result) {
    x_.resize(sz_);
    y_.resize(sz_);

    src_host_t xh_(sz_, 0);
    dst_host_t yh_(sz_, 0);

    result = 0;
    for (int i = 0; i < sz_; i++) {
      xh_(i) = i + 1;
      yh_(i) = sz_ - i;
      result += (i + 1) * (sz_ - i);
    }

    Morpheus::copy(xh_, x_);
    Morpheus::copy(yh_, y_);
  }
};

namespace Test {

#define MORPHEUS_VALIDATE_DOT(_value_type, _res, _ref_res)                   \
  {                                                                          \
    /* Make sure the correct type is returned by dot */                      \
    EXPECT_EQ((std::is_same<decltype(_res), decltype(_ref_res)>::value), 1); \
    if (std::is_floating_point<_value_type>::value) {                        \
      EXPECT_PRED_FORMAT2(                                                   \
          ::testing::internal::CmpHelperFloatingPointEQ<_value_type>,        \
          _ref_res, _res);                                                   \
    } else {                                                                 \
      EXPECT_EQ(_ref_res, _res);                                             \
    }                                                                        \
  }

TYPED_TEST_SUITE(DotTypesTest, DotTypes);

TYPED_TEST(DotTypesTest, DotCustom) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  // Small Custom
  {
    index_type sz      = this->small_size;
    value_type ref_res = this->small_ref_res;
    auto x = this->small_x, y = this->small_y;

    auto result = Morpheus::dot<TEST_CUSTOM_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }

  // Medium Custom
  {
    index_type sz      = this->med_size;
    value_type ref_res = this->med_ref_res;
    auto x = this->med_x, y = this->med_y;

    auto result = Morpheus::dot<TEST_CUSTOM_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }

  // Large Custom
  {
    index_type sz      = this->large_size;
    value_type ref_res = this->large_ref_res;
    auto x = this->large_x, y = this->large_y;

    auto result = Morpheus::dot<TEST_CUSTOM_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }
}

TYPED_TEST(DotTypesTest, DotGeneric) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  // Small Generic
  {
    index_type sz      = this->small_size;
    value_type ref_res = this->small_ref_res;
    auto x = this->small_x, y = this->small_y;

    auto result = Morpheus::dot<TEST_GENERIC_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }

  // Medium Generic
  {
    index_type sz      = this->med_size;
    value_type ref_res = this->med_ref_res;
    auto x = this->med_x, y = this->med_y;

    auto result = Morpheus::dot<TEST_GENERIC_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }

  // Large Generic
  {
    index_type sz      = this->large_size;
    value_type ref_res = this->large_ref_res;
    auto x = this->large_x, y = this->large_y;

    auto result = Morpheus::dot<TEST_GENERIC_SPACE>(sz, x, y);
    MORPHEUS_VALIDATE_DOT(value_type, result, ref_res);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DOT_HPP
