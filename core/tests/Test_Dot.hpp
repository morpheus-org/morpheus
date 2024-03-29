/**
 * Test_Dot.hpp
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
  using IndexType  = size_t;
  using ValueType  = typename dst_dev_t::value_type;

  struct vectors {
    src_dev_t x;
    dst_dev_t y;
    IndexType size;
    ValueType dot;

    vectors(IndexType _size)
        : x(_size, 0), y(_size, 0), size(_size), dot((ValueType)0) {}
  };

  IndexType sizes[3] = {50, 5000, 50000};

  struct vectors vecs[3] = {vectors(sizes[0]), vectors(sizes[1]),
                            vectors(sizes[2])};

  void SetUp() override {
    for (size_t i = 0; i < 3; i++) {
      local_setup(&vecs[i]);
    }
  }

 private:
  void local_setup(struct vectors* vec) {
    src_host_t xh_(vec->size, 0);
    dst_host_t yh_(vec->size, 0);

    vec->dot = 0;
    for (IndexType i = 0; i < vec->size; i++) {
      xh_(i) = i + 1;
      yh_(i) = vec->size - i;
      vec->dot += (i + 1) * (vec->size - i);
    }

    Morpheus::copy(xh_, vec->x);
    Morpheus::copy(yh_, vec->y);
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

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::dot<TEST_CUSTOM_SPACE>(v.size, v.x, v.y);
    MORPHEUS_VALIDATE_DOT(value_type, result, v.dot);
  }
}

TYPED_TEST(DotTypesTest, DotGeneric) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::dot<TEST_GENERIC_SPACE>(v.size, v.x, v.y);
    MORPHEUS_VALIDATE_DOT(value_type, result, v.dot);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DOT_HPP
