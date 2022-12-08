/**
 * Test_WAXPBY.hpp
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

#ifndef TEST_CORE_TEST_WAXPBY_HPP
#define TEST_CORE_TEST_WAXPBY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using DenseVectorPairs =
    generate_pair<generate_pair<DenseVectorTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;

using WAXPBYTypes = to_gtest_types<DenseVectorPairs>::type;

template <typename Containers>
class WAXPBYTypesTest : public ::testing::Test {
 public:
  using type             = Containers;
  using src1_container_t = typename Containers::first_type::first_type::type;
  using src2_container_t = typename Containers::first_type::second_type::type;
  using dst_container_t  = typename Containers::second_type::type;
  using src1_dev_t       = typename src1_container_t::type;
  using src1_host_t      = typename src1_container_t::type::HostMirror;
  using src2_dev_t       = typename src2_container_t::type;
  using src2_host_t      = typename src2_container_t::type::HostMirror;
  using dst_dev_t        = typename dst_container_t::type;
  using dst_host_t       = typename dst_container_t::type::HostMirror;
  using IndexType        = size_t;
  using ValueType1       = typename src1_dev_t::value_type;
  using ValueType2       = typename src2_dev_t::value_type;
  using ValueType3       = typename dst_dev_t::value_type;

  src1_dev_t small_x, med_x, large_x;
  src2_dev_t small_y, med_y, large_y;
  dst_dev_t small_w, med_w, large_w;

  IndexType small_size = 50, med_size = 5000, large_size = 50000;
  ValueType1 alpha = 0.15;
  ValueType2 beta  = -1.2;

  // You can define per-test set-up logic as usual.
  void SetUp() override {
    local_setup(small_size, small_x, small_y, small_w);
    local_setup(med_size, med_x, med_y, med_w);
    local_setup(large_size, large_x, large_y, large_w);
  }

 private:
  void local_setup(IndexType sz_, src1_dev_t& x_, src2_dev_t& y_,
                   dst_dev_t& w_) {
    x_.resize(sz_);
    y_.resize(sz_);
    w_.resize(sz_);

    src1_host_t xh_(sz_, 0);
    src2_host_t yh_(sz_, 0);
    dst_host_t wh_(sz_, 0);

    for (IndexType i = 0; i < sz_; i++) {
      xh_(i) = i + 1;
      yh_(i) = sz_ - i;
      wh_(i) = alpha * xh_(i) + beta * yh_(i);
    }

    Morpheus::copy(xh_, x_);
    Morpheus::copy(yh_, y_);
    Morpheus::copy(wh_, w_);
  }
};

namespace Test {

TYPED_TEST_SUITE(WAXPBYTypesTest, WAXPBYTypes);

TYPED_TEST(WAXPBYTypesTest, WAXPBYCustom) {
  using dst_t      = typename TestFixture::dst_dev_t;
  using index_type = typename TestFixture::IndexType;

  auto a = this->alpha;
  auto b = this->beta;

  // Small Custom
  {
    index_type sz = this->small_size;
    auto x        = this->small_x;
    auto y        = this->small_y;
    auto wref     = this->small_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_CUSTOM_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }

  // Medium Custom
  {
    index_type sz = this->med_size;
    auto x        = this->med_x;
    auto y        = this->med_y;
    auto wref     = this->med_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_CUSTOM_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }

  // Large Custom
  {
    index_type sz = this->large_size;
    auto x        = this->large_x;
    auto y        = this->large_y;
    auto wref     = this->large_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_CUSTOM_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }
}

TYPED_TEST(WAXPBYTypesTest, WAXPBYGeneric) {
  using dst_t      = typename TestFixture::dst_dev_t;
  using index_type = typename TestFixture::IndexType;

  auto a = this->alpha;
  auto b = this->beta;

  // Small Custom
  {
    index_type sz = this->small_size;
    auto x        = this->small_x;
    auto y        = this->small_y;
    auto wref     = this->small_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_GENERIC_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }

  // Medium Custom
  {
    index_type sz = this->med_size;
    auto x        = this->med_x;
    auto y        = this->med_y;
    auto wref     = this->med_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_GENERIC_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }

  // Large Custom
  {
    index_type sz = this->large_size;
    auto x        = this->large_x;
    auto y        = this->large_y;
    auto wref     = this->large_w;

    dst_t w(sz, 0);

    Morpheus::waxpby<TEST_GENERIC_SPACE>(sz, a, x, b, y, w);
    Morpheus::Test::have_same_data(w, wref);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_WAXPBY_HPP
