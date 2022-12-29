/**
 * Test_MirrorContainers.hpp
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

#ifndef TEST_CORE_TEST_MIRRORCONTAINERS_HPP
#define TEST_CORE_TEST_MIRRORCONTAINERS_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>
#include <utils/Macros_DenseMatrix.hpp>
#include <utils/Macros_DenseVector.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

// Generate all unary combinations for every container and combine
using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;

using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types::types_set>::type;

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::types_set>::type;

using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types::types_set>::type;

using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::types_set>::type;
using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;

using ContainerTypes = Morpheus::concat<
    DenseVectorTypes,
    Morpheus::concat<
        DenseMatrixTypes,
        Morpheus::concat<CooMatrixTypes,
                         Morpheus::concat<CsrMatrixTypes,
                                          Morpheus::concat<DiaMatrixTypes,
                                                           DynamicMatrixTypes>::
                                              type>::type>::type>::type>::type;
using ContainerTypesUnary = to_gtest_types<ContainerTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class ContainerTypesUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;

  ContainerTypesUnaryTest() : Aref(), Ahref() {}

  void SetUp() override {
    Morpheus::Test::setup_small_container(Ahref);
    Aref.resize(Ahref);
    Morpheus::copy(Ahref, Aref);
  }

  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary Container Types
 *
 */
TYPED_TEST_SUITE(ContainerTypesUnaryTest, ContainerTypesUnary);

TYPED_TEST(ContainerTypesUnaryTest, Mirror_SameSpace) {
  using HostContainer = typename TestFixture::host;

  auto c   = Morpheus::create_mirror(this->Aref);
  bool res = std::is_same_v<decltype(c), HostContainer>;
  EXPECT_EQ(res, 1);

  // Check shape same to Aref
  bool same_size = Morpheus::Test::is_same_size(c, this->Aref);
  EXPECT_EQ(same_size, 1);

  // Check c is empty
  bool empty = Morpheus::Test::is_empty_container(c);
  EXPECT_EQ(empty, 1);
}

TYPED_TEST(ContainerTypesUnaryTest, Mirror_NewSpace) {
  using Container = typename TestFixture::device;

  using space        = TEST_CUSTOM_SPACE;
  using value_type   = typename Container::non_const_value_type;
  using index_type   = typename Container::non_const_index_type;
  using array_layout = typename Container::array_layout;

  auto c = Morpheus::create_mirror<space>(this->Aref);

  bool res =
      std::is_same_v<decltype(c), decltype(Morpheus::Test::create_container<
                                           Container, value_type, index_type,
                                           array_layout, space>())>;
  EXPECT_EQ(res, 1);

  // Check shape same to Aref
  bool same_size = Morpheus::Test::is_same_size(c, this->Aref);
  EXPECT_EQ(same_size, 1);

  // Check c is empty
  bool empty = Morpheus::Test::is_empty_container(c);
  EXPECT_EQ(empty, 1);
}

TYPED_TEST(ContainerTypesUnaryTest, MirrorContainer_SameSpace) {
  using Container     = typename TestFixture::device;
  using HostContainer = typename TestFixture::host;

  auto c = Morpheus::create_mirror_container(this->Aref);

  // Check mirror container same type as HostMirror
  bool res = std::is_same_v<decltype(c), HostContainer>;
  EXPECT_EQ(res, 1);

  // Check shape same to Aref
  bool same_size = Morpheus::Test::is_same_size(c, this->Aref);
  EXPECT_EQ(same_size, 1);

  if (Morpheus::is_compatible_v<Container, HostContainer>) {
    // new container should have same data as device
    bool same_data = Morpheus::Test::have_same_data(c, this->Aref);
    EXPECT_EQ(same_data, 1);
  } else {
    // new allocation
    bool empty = Morpheus::Test::is_empty_container(c);
    EXPECT_EQ(empty, 1);
  }
}

TYPED_TEST(ContainerTypesUnaryTest, MirrorContainer_NewSpace) {
  using Container = typename TestFixture::device;

  using space        = TEST_CUSTOM_SPACE;
  using value_type   = typename Container::non_const_value_type;
  using index_type   = typename Container::non_const_index_type;
  using array_layout = typename Container::array_layout;

  auto c = Morpheus::create_mirror_container<space>(this->Aref);

  // Check shape same to Aref
  bool same_size = Morpheus::Test::is_same_size(c, this->Aref);
  EXPECT_EQ(same_size, 1);

  if (Morpheus::has_same_memory_space_v<Container, space>) {
    // Check mirror container has same type as Original Container
    bool res = std::is_same_v<decltype(c), Container>;
    EXPECT_EQ(res, 1);

    // new container should have same data as device
    bool same_data = Morpheus::Test::have_same_data(c, this->Aref);
    EXPECT_EQ(same_data, 1);
  } else {
    // Check mirror container has same type as Original Container in new space
    bool res =
        std::is_same_v<decltype(c), decltype(Morpheus::Test::create_container<
                                             Container, value_type, index_type,
                                             array_layout, space>())>;
    EXPECT_EQ(res, 1);

    // new allocation
    bool empty = Morpheus::Test::is_empty_container(c);
    EXPECT_EQ(empty, 1);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_MIRRORCONTAINERS_HPP