/**
 * Test_DenseVector.hpp
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

#ifndef TEST_CORE_TEST_DENSEVECTOR_HPP
#define TEST_CORE_TEST_DENSEVECTOR_HPP

#include <Morpheus_Core.hpp>

#include <setup/DenseVectorDefinition_Utils.hpp>

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DenseVectorUnaryTest : public ::testing::Test {
 public:
  using DenseVector = UnaryContainer;
  using HostMirror  = typename UnaryContainer::HostMirror;
  // No need for setup and tear-down in this case, mainly care about the types
  // any setup and tear-down will be made by each individual test
};

// Used for testing behaviour between many types of the same container
template <typename DenseVectorTypes>
class DenseVectorTypesTest : public ::testing::Test {
 public:
  using DenseVector_v    = typename DenseVectorTypes::v;
  using DenseVector_vl   = typename DenseVectorTypes::vl;
  using DenseVector_vis  = typename DenseVectorTypes::vis;
  using DenseVector_vil  = typename DenseVectorTypes::vil;
  using DenseVector_vils = typename DenseVectorTypes::vils;
  using DenseVector_vls  = typename DenseVectorTypes::vls;
};

namespace Test {

TYPED_TEST_CASE(DenseVectorUnaryTest, DenseVectorUnary);

TYPED_TEST(DenseVectorUnaryTest, Traits) {
  // Check DenseVector Specific Traits:
  // Tag, value_array_type, value_array_pointer, value_array_reference
  // Repeat that for the HostMirror too
  // Check value_array_type traits too
  // Ensure size is of type size_t and not index_type
  // Add size_type trait
  static_assert(std::is_same<typename TestFixture::DenseVector::tag,
                             Morpheus::DenseVectorTag>::value);

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, DefaultConstruction) {
  // DenseVector()

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, NormalConstruction) {
  // DenseVector(const std::string name, index_type n, value_type val = 0)
  // DenseVector(index_type n, value_type val = 0)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, RandomConstruction) {
  // DenseVector(const std::string name, index_type n, Generator rand_pool,
  //             const value_type range_low, const value_type range_high)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, ShallowCopySemantics) {
  // DenseVector(const DenseVector<VR, PR...>& src, ...)
  // operator=(const DenseVector<VR, PR...>& src)
  // DenseVector(const DenseVector&) = default;
  // DenseVector& operator=(const DenseVector&) = default;

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, Allocate) {
  // DenseVector& allocate(const std::string name,
  //                       const DenseVector<VR, PR...>& src)

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, Assign) {
  // assign(const index_type n, const value_type val)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, AssignRandom) {
  // assign(Generator rand_pool, const value_type range_low,
  //        const value_type range_high)

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, Resize) {
  // resize(index_type n)
  // resize(const index_type n, const index_type val)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorUnaryTest, UtilRoutines) {
  // size()
  // data()
  // view()
  // const_view()
  // name()

  FAIL();
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_HPP