/**
 * Test_CooMatrix.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_HPP
#define TEST_CORE_TEST_COOMATRIX_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::test_types_set>::type;
using CooMatrixUnary = to_gtest_types<CooMatrixTypes>::type;

using CooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixTypes, CooMatrixTypes>::type>::type;

using CooMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CooMatrix<double>, types::compatible_types_set>::type;

using CompatibleCooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixCompatibleTypes, CooMatrixCompatibleTypes>::type>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class CooMatrixUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;
};

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleCooMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CooMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CooMatrix
  using host2   = typename type2::type::HostMirror;
};

// Used for testing binary operations
template <typename BinaryContainer>
class CooMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CooMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CooMatrix
  using host2   = typename type2::type::HostMirror;
};

/**
 * @brief Test Suite using the Unary CooMatrix
 *
 */
TYPED_TEST_CASE(CooMatrixUnaryTest, CooMatrixUnary);

/**
 * @brief Testing default construction of CooMatrix container
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultConstruction) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, FormatEnum) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, FormatIndex) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, ReferenceByIndex) {
  // row_indices
  // column_indices
  // values
  // crow_indices
  // ccolumn_indices
  // cvalues
  EXPECT_EQ(1, 0);
}

TYPED_TEST(CooMatrixUnaryTest, Reference) {
  // row_indices
  // column_indices
  // values
  // crow_indices
  // ccolumn_indices
  // cvalues
  EXPECT_EQ(1, 0);
}

TYPED_TEST(CooMatrixUnaryTest, DefaultCopyAssignment) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, DefaultCopyConstructor) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, DefaultMoveAssignment) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, DefaultMoveConstructor) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, ConstructionFromShape) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, Resize) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, SortByRow) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, SortByRowAndColumn) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, IsSortedByRow) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixUnaryTest, IsSorted) { EXPECT_EQ(1, 0); }

/**
 * @brief Test Suite using the Compatible Binary CooMatrix pairs
 *
 */
TYPED_TEST_CASE(CompatibleCooMatrixBinaryTest, CompatibleCooMatrixBinary);

TYPED_TEST(CompatibleCooMatrixBinaryTest, ConstructionFromCooMatrix) {
  EXPECT_EQ(1, 0);
}

TYPED_TEST(CompatibleCooMatrixBinaryTest, CopyAssignmentFromCooMatrix) {
  EXPECT_EQ(1, 0);
}

/**
 * @brief Test Suite using the Binary CooMatrix pairs
 *
 */
TYPED_TEST_CASE(CooMatrixBinaryTest, CooMatrixBinary);

TYPED_TEST(CooMatrixBinaryTest, ResizeFromCooMatrix) { EXPECT_EQ(1, 0); }

TYPED_TEST(CooMatrixBinaryTest, AllocateFromCooMatrix) { EXPECT_EQ(1, 0); }

/**
 * @brief Test Suite using the Binary CooMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_CASE(CompatibleCooMatrixDynamicTest, CooMatrixDynamic);

TYPED_TEST(CompatibleCooMatrixDynamicTest, ConstructionFromDynamicMatrix) {
  EXPECT_EQ(1, 0);
}

TYPED_TEST(CompatibleCooMatrixDynamicTest, CopyAssignmentFromDynamicMatrix) {
  EXPECT_EQ(1, 0);
}

#endif  // TEST_CORE_TEST_COOMATRIX_HPP