/**
 * Test_Reduce.hpp
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

#ifndef TEST_CORE_TEST_REDUCE_HPP
#define TEST_CORE_TEST_REDUCE_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;

using ReduceTypes = to_gtest_types<DenseVectorTypes>::type;

template <typename UnaryContainer>
class ReduceTypesTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using dev_t  = typename UnaryContainer::type;
  using host_t = typename UnaryContainer::type::HostMirror;

  using IndexType = typename dev_t::index_type;
  using ValueType = typename dev_t::value_type;

  struct vector {
    dev_t data;
    IndexType size;
    ValueType reduction;

    vector(IndexType _vec_size, IndexType _reduction_size)
        : data(_vec_size, 0), size(_reduction_size), reduction((ValueType)0) {}
  };

  IndexType vec_sizes[6] = {50, 50, 5000, 5000, 50000, 50000};
  IndexType red_sizes[6] = {50, 25, 5000, 1000, 50000, 100};

  struct vector vecs[6] = {
      vector(vec_sizes[0], red_sizes[0]), vector(vec_sizes[1], red_sizes[1]),
      vector(vec_sizes[2], red_sizes[2]), vector(vec_sizes[3], red_sizes[3]),
      vector(vec_sizes[4], red_sizes[4]), vector(vec_sizes[5], red_sizes[5])};

  void SetUp() override {
    for (size_t i = 0; i < 6; i++) {
      local_setup(&vecs[i]);
    }
  }

 private:
  void local_setup(struct vector* vec) {
    host_t data_h(vec->data.size(), 0);

    for (size_t i = 0; i < vec->data.size(); i++) {
      data_h(i) = i + (ValueType)1.5;
    }

    vec->reduction = 0;
    for (IndexType i = 0; i < vec->size; i++) {
      vec->reduction += data_h(i);
    }

    Morpheus::copy(data_h, vec->data);
  }
};

namespace Test {

#define MORPHEUS_VALIDATE_REDUCE(_value_type, _res, _ref_res)                \
  {                                                                          \
    /* Make sure the correct type is returned by reduce */                   \
    EXPECT_EQ((std::is_same<decltype(_res), decltype(_ref_res)>::value), 1); \
    if (std::is_floating_point<_value_type>::value) {                        \
      EXPECT_PRED_FORMAT2(                                                   \
          ::testing::internal::CmpHelperFloatingPointEQ<_value_type>,        \
          _ref_res, _res);                                                   \
    } else {                                                                 \
      EXPECT_EQ(_ref_res, _res);                                             \
    }                                                                        \
  }

TYPED_TEST_SUITE(ReduceTypesTest, ReduceTypes);

TYPED_TEST(ReduceTypesTest, ReduceCustom) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 6; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::reduce<TEST_CUSTOM_SPACE>(v.data, v.size);
    MORPHEUS_VALIDATE_REDUCE(value_type, result, v.reduction);
  }
}

TYPED_TEST(ReduceTypesTest, ReduceGeneric) {
  using value_type = typename TestFixture::ValueType;
  using index_type = typename TestFixture::IndexType;

  for (index_type i = 0; i < 6; i++) {
    auto v = this->vecs[i];

    auto result = Morpheus::reduce<TEST_GENERIC_SPACE>(v.data, v.size);
    MORPHEUS_VALIDATE_REDUCE(value_type, result, v.reduction);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_REDUCE_HPP
