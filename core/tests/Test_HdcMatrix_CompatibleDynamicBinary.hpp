/**
 * Test_HdcMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_HDCMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_HDCMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HdcMatrix.hpp>

using HdcMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::HdcMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using HdcMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HdcMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and HdcMatrix
// containers
template <typename BinaryContainer>
class CompatibleHdcMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // HdcMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  using SizeType = typename device::size_type;

  CompatibleHdcMatrixDynamicTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        dia_nnnz(SMALL_HDC_DIA_NNZ),
        csr_nnnz(SMALL_HDC_CSR_NNZ),
        ndiag(SMALL_HDC_DIA_NDIAG),
        nalign(SMALL_MATRIX_ALIGNMENT),
        Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HDC_DIA_NNZ,
             SMALL_HDC_CSR_NNZ, SMALL_HDC_DIA_NDIAG, SMALL_MATRIX_ALIGNMENT),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HDC_DIA_NNZ,
              SMALL_HDC_CSR_NNZ, SMALL_HDC_DIA_NDIAG, SMALL_MATRIX_ALIGNMENT) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  SizeType nrows, ncols, dia_nnnz, csr_nnnz, ndiag, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary HdcMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleHdcMatrixDynamicTest,
                 HdcMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of HdcMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the HdcMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleHdcMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveHdc) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference HdcMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is HdcMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_HDC_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.dia().diagonal_offsets(1) = 2;
  Bh.dia().values(1, 0)        = (value_type)-1.11;
  Bh.csr().row_offsets(0)      = 1;
  Bh.csr().column_indices(0)   = 1;
  Bh.csr().values(0)           = (value_type)-1.11;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), Ch.dia().diagonal_offsets(i));
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), Ch.dia().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), Ch.csr().row_offsets(i));
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), Ch.csr().column_indices(i));
    EXPECT_EQ(Bh.csr().values(i), Ch.csr().values(i));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_HDC_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                this->ndiag);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), Ct.dia().diagonal_offsets(i));
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), Ct.dia().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), Ct.csr().row_offsets(i));
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), Ct.csr().column_indices(i));
    EXPECT_EQ(Bh.csr().values(i), Ct.csr().values(i));
  }
}

/**
 * @brief Testing construction of HdcMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the HdcMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleHdcMatrixDynamicTest,
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
 * @brief Testing copy assignment of HdcMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the HdcMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleHdcMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveHdc) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference HdcMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is HdcMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignment from dynamic
  HostMatrix Bh = Ah;
  CHECK_HDC_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.dia().diagonal_offsets(1) = 2;
  Bh.dia().values(1, 0)        = (value_type)-1.11;
  Bh.csr().row_offsets(0)      = 1;
  Bh.csr().column_indices(0)   = 1;
  Bh.csr().values(0)           = (value_type)-1.11;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), Ch.dia().diagonal_offsets(i));
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), Ch.dia().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), Ch.csr().row_offsets(i));
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), Ch.csr().column_indices(i));
    EXPECT_EQ(Bh.csr().values(i), Ch.csr().values(i));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_HDC_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                this->ndiag);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), Ct.dia().diagonal_offsets(i));
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), Ct.dia().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), Ct.csr().row_offsets(i));
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), Ct.csr().column_indices(i));
    EXPECT_EQ(Bh.csr().values(i), Ct.csr().values(i));
  }
}

/**
 * @brief Testing copy assignment of HdcMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the HdcMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleHdcMatrixDynamicTest,
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
#endif  // TEST_CORE_TEST_HDCMATRIX_COMPATIBLEDYNAMICBINARY_HPP