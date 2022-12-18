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
  using SizeType         = typename src1_dev_t::size_type;
  using ValueType1       = typename src1_dev_t::value_type;
  using ValueType2       = typename src2_dev_t::value_type;
  using ValueType3       = typename dst_dev_t::value_type;

  struct vectors {
    src1_dev_t x;
    src2_dev_t y;
    dst_dev_t w;
    SizeType size;
    ValueType1 alpha;
    ValueType2 beta;

    vectors(SizeType _size, ValueType1 _alpha, ValueType2 _beta)
        : x(_size, 0),
          y(_size, 0),
          w(_size, 0),
          size(_size),
          alpha(_alpha),
          beta(_beta) {}
  };

  SizeType sizes[3]    = {50, 5000, 50000};
  ValueType1 alphas[3] = {(ValueType1)2.15, (ValueType1)2.15, (ValueType1)2.15};
  ValueType2 betas[3]  = {(ValueType2)-3.2, (ValueType2)-3.2, (ValueType2)-3.2};

  struct vectors vecs[3] = {vectors(sizes[0], alphas[0], betas[0]),
                            vectors(sizes[1], alphas[1], betas[1]),
                            vectors(sizes[2], alphas[2], betas[2])};

  void SetUp() override {
    for (SizeType i = 0; i < 3; i++) {
      local_setup(&vecs[i]);
    }
  }

 private:
  void local_setup(struct vectors* vec) {
    src1_host_t xh_(vec->size, 0);
    src2_host_t yh_(vec->size, 0);
    dst_host_t wh_(vec->size, 0);

    for (SizeType i = 0; i < vec->size; i++) {
      xh_(i) = -5.0 + (((ValueType1)rand() / RAND_MAX) * (5.0 - (-5.0)));
      yh_(i) = -2.0 + (((ValueType2)rand() / RAND_MAX) * (2.0 - (-2.0)));
      wh_(i) = vec->alpha * xh_(i) + vec->beta * yh_(i);
    }

    Morpheus::copy(xh_, vec->x);
    Morpheus::copy(yh_, vec->y);
    Morpheus::copy(wh_, vec->w);
  }
};

namespace Test {

TYPED_TEST_SUITE(WAXPBYTypesTest, WAXPBYTypes);

TYPED_TEST(WAXPBYTypesTest, WAXPBYCustom) {
  using dst_t     = typename TestFixture::dst_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    dst_t w(v.size, 0);

    Morpheus::waxpby<TEST_CUSTOM_SPACE>(v.size, v.alpha, v.x, v.beta, v.y, w);
    EXPECT_EQ(Morpheus::Test::is_empty_container(w), 0);
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(w, v.w));
  }
}

TYPED_TEST(WAXPBYTypesTest, WAXPBYGeneric) {
  using dst_t     = typename TestFixture::dst_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    dst_t w(v.size, 0);

    Morpheus::waxpby<TEST_GENERIC_SPACE>(v.size, v.alpha, v.x, v.beta, v.y, w);
    EXPECT_EQ(Morpheus::Test::is_empty_container(w), 0);
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(w, v.w));
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_WAXPBY_HPP
