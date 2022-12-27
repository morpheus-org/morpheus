/**
 * Test_CsrMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_CSRMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_CSRMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_CsrMatrix.hpp>

using CsrMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CsrMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using CsrMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CsrMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and CsrMatrix
// containers
template <typename BinaryContainer>
class CompatibleCsrMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // CsrMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  CompatibleCsrMatrixDynamicTest()
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
 * @brief Test Suite using the Binary CsrMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleCsrMatrixDynamicTest,
                 CsrMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of CsrMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the CsrMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleCsrMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveCsr) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz = SMALL_MATRIX_NNZ;
  // Build matrix from the reference CsrMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CsrMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_CSR_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.row_offsets(2)    = 6;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), Ch.row_offsets(n));
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_CSR_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), Ct.row_offsets(n));
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing construction of CsrMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the CsrMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleCsrMatrixDynamicTest,
           ConstructionFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B(A), Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh(Ah), Morpheus::RuntimeException);
}

/**
 * @brief Testing copy assignment of CsrMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the CsrMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleCsrMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveCsr) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz = SMALL_MATRIX_NNZ;
  // Build matrix from the reference CsrMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CsrMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignemnt from dynamic
  HostMatrix Bh = Ah;
  CHECK_CSR_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.row_offsets(2)    = 6;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), Ch.row_offsets(n));
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_CSR_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), Ct.row_offsets(n));
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing copy assignment of CsrMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the CsrMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleCsrMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B = A, Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh = Ah, Morpheus::RuntimeException);
}

}  // namespace Test
#endif  // TEST_CORE_TEST_CSRMATRIX_COMPATIBLEDYNAMICBINARY_HPP