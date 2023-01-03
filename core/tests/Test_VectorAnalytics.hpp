/**
 * Test_VectorAnalytics.hpp
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

#ifndef TEST_CORE_TEST_VECTORANALYTICS_HPP
#define TEST_CORE_TEST_VECTORANALYTICS_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using VectorAnalyticsTypes = to_gtest_types<DenseVectorTypes>::type;

template <typename Containers>
class VectorAnalyticsTypesTest : public ::testing::Test {
 public:
  using type       = Containers;
  using src_t      = typename Containers::type;
  using src_dev_t  = typename src_t::type;
  using src_host_t = typename src_t::type::HostMirror;
  using IndexType  = size_t;
  using ValueType  = typename src_t::value_type;

  struct vectors {
    src_dev_t v;
    IndexType size;
    ValueType min, max;

    vectors(IndexType _size)
        : v(_size, 0), size(_size), min((ValueType)0), max((ValueType)0) {}
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
    src_host_t vh_(vec->size, 0);

    vec->min = 0;
    vec->max = 0;

    unsigned long long seed = 5374857;
    Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);
    vh_.assign(vh_.size(), rand_pool, -5.0, 5.0);
    for (IndexType i = 0; i < vec->size; i++) {
      vec->min = vec->min < vh_(i) ? vec->min : vh_(i);
      vec->max = vec->max > vh_(i) ? vec->max : vh_(i);
    }

    Morpheus::copy(vh_, vec->v);
  }
};

namespace Test {

#define MORPHEUS_VALIDATE_MINMAX(_value_type, _res, _ref_res)                \
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

TYPED_TEST_SUITE(VectorAnalyticsTypesTest, VectorAnalyticsTypes);

TYPED_TEST(VectorAnalyticsTypesTest, MaxCustom) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::max<TEST_CUSTOM_SPACE>(v.v, v.size);
    MORPHEUS_VALIDATE_MINMAX(value_type, result, v.max);
  }
}

TYPED_TEST(VectorAnalyticsTypesTest, MaxGeneric) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::max<TEST_GENERIC_SPACE>(v.v, v.size);
    MORPHEUS_VALIDATE_MINMAX(value_type, result, v.max);
  }
}

TYPED_TEST(VectorAnalyticsTypesTest, MinCustom) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::min<TEST_CUSTOM_SPACE>(v.v, v.size);
    MORPHEUS_VALIDATE_MINMAX(value_type, result, v.min);
  }
}

TYPED_TEST(VectorAnalyticsTypesTest, MinGeneric) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 3; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::min<TEST_GENERIC_SPACE>(v.v, v.size);
    MORPHEUS_VALIDATE_MINMAX(value_type, result, v.min);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_VECTORANALYTICS_HPP
