/**
 * Test_CooMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_CooMatrix.hpp>

using CooMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CooMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using CooMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and CooMatrix
// containers
template <typename BinaryContainer>
class CompatibleCooMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // CooMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  CompatibleCooMatrixDynamicTest()
      : Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary CooMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleCooMatrixDynamicTest,
                 CooMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of CooMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the CooMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveCoo) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz = SMALL_MATRIX_NNZ;
  // Build matrix from the reference CooMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CooMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_COO_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.row_indices(2)    = 2;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ch.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_COO_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ct.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing construction of CooMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the CooMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           ConstructionFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B(A), Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh(Ah), Morpheus::RuntimeException);
}

/**
 * @brief Testing copy assignment of CooMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the CooMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveCoo) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz = SMALL_MATRIX_NNZ;
  // Build matrix from the reference CooMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CooMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignemnt from dynamic
  HostMatrix Bh = Ah;
  CHECK_COO_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.row_indices(2)    = 2;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ch.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_COO_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ct.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing copy assignment of CooMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the CooMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B = A, Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh = Ah, Morpheus::RuntimeException);
}

}  // namespace Test
#endif  // TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP