/**
 * Test_HdcMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_HDCMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_HDCMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HdcMatrix.hpp>

using HdcMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::HdcMatrix<double>, types::compatible_types_set>::type;

using CompatibleHdcMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HdcMatrixCompatibleTypes, HdcMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleHdcMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // HdcMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // HdcMatrix
  using host2   = typename type2::type::HostMirror;

  using SizeType = typename device1::size_type;

  CompatibleHdcMatrixBinaryTest()
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
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Compatible Binary HdcMatrix pairs
 *
 */
TYPED_TEST_SUITE(CompatibleHdcMatrixBinaryTest, CompatibleHdcMatrixBinary);

TYPED_TEST(CompatibleHdcMatrixBinaryTest, ConstructionFromHdcMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                 this->ndiag, this->nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type)-1.11;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                 this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

TYPED_TEST(CompatibleHdcMatrixBinaryTest, CopyAssignmentFromHdcMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                 this->ndiag, this->nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh = Ah;
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type)-1.11;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                 this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing construction of HdcMatrix from \p CsrMatrix and \p DiaMatrix.
 *
 */
TYPED_TEST(CompatibleHdcMatrixBinaryTest, ConstructionFromDenseMatrix) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                this->ndiag, this->nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  HostMatrix Ah_test(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                     this->ndiag, this->nalign);
  Morpheus::copy(A, Ah_test);

  VALIDATE_HDC_CONTAINER(Ah, Ah_test);

  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type)-1.11;
  EXPECT_NE(Ah.dia().diagonal_offsets(1), Ah_test.dia().diagonal_offsets(1));
  EXPECT_NE(Ah.dia().values(1, 0), Ah_test.dia().values(1, 0));
  EXPECT_NE(Ah.csr().row_offsets(0), Ah_test.csr().row_offsets(0));
  EXPECT_NE(Ah.csr().column_indices(0), Ah_test.csr().column_indices(0));
  EXPECT_NE(Ah.csr().values(0), Ah_test.csr().values(0));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_HDCMATRIX_COMPATIBLEBINARY_HPP