/**
 * Test_HdcMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_HDCMATRIX_BINARY_HPP
#define TEST_CORE_TEST_HDCMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HdcMatrix.hpp>

using HdcMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HdcMatrix<double>,
                                               types::types_set>::type;
using HdcMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HdcMatrixTypes, HdcMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class HdcMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // HdcMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // HdcMatrix
  using host2   = typename type2::type::HostMirror;

  using IndexType = typename device1::size_type;

  HdcMatrixBinaryTest()
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

  IndexType nrows, ncols, dia_nnnz, csr_nnnz, ndiag, nalign;
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary HdcMatrix pairs
 *
 */
TYPED_TEST_SUITE(HdcMatrixBinaryTest, HdcMatrixBinary);

TYPED_TEST(HdcMatrixBinaryTest, ResizeFromHdcMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using size_type   = typename Matrix1::size_type;
  using value_type  = typename Matrix1::value_type;

  Matrix1 A(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
            this->ndiag);
  CHECK_HDC_SIZES(A, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  size_type large_nrows = 500, large_ncols = 500, large_dia_nnnz = 640,
            large_csr_nnnz = 340, large_ndiag = 110;
  Matrix2 Alarge(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                 large_ndiag);
  A.resize(Alarge);
  CHECK_HDC_CONTAINERS(Alarge, A);

  HostMatrix1 Ah(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                 large_ndiag);
  CHECK_HDC_SIZES(Ah, large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                  large_ndiag, this->nalign);

  Morpheus::copy(A, Ah);

  for (size_type i = 0; i < this->Ahref.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Ah.dia().diagonal_offsets(i),
              this->Ahref.dia().diagonal_offsets(i));
  }

  for (size_type i = 0; i < this->Ahref.dia().values().nrows(); i++) {
    for (size_type j = 0; j < this->Ahref.dia().values().ncols(); j++) {
      EXPECT_EQ(Ah.dia().values(i, j), this->Ahref.dia().values(i, j));
    }
  }

  for (size_type i = this->Ahref.dia().diagonal_offsets().size();
       i < Ah.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Ah.dia().diagonal_offsets(i), 0);
  }

  for (size_type i = this->Ahref.dia().values().nrows();
       i < Ah.dia().values().nrows(); i++) {
    for (size_type j = this->Ahref.dia().values().ncols();
         j < Ah.dia().values().ncols(); j++) {
      EXPECT_EQ(Ah.dia().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < this->Ahref.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Ah.csr().row_offsets(i), this->Ahref.csr().row_offsets(i));
  }

  for (size_type i = 0; i < this->Ahref.csr().values().size(); i++) {
    EXPECT_EQ(Ah.csr().column_indices(i), this->Ahref.csr().column_indices(i));
    EXPECT_EQ(Ah.csr().values(i), this->Ahref.csr().values(i));
  }

  for (size_type i = this->Ahref.csr().row_offsets().size();
       i < Ah.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Ah.csr().row_offsets(i), 0);
  }

  for (size_type i = this->Ahref.csr().values().size();
       i < Ah.csr().values().size(); i++) {
    EXPECT_EQ(Ah.csr().column_indices(i), 0);
    EXPECT_EQ(Ah.csr().values(i), 0);
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix1 Ahref_test(this->nrows, this->ncols, this->dia_nnnz,
                         this->csr_nnnz, this->ndiag);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.dia().diagonal_offsets(1), Ahref_test.dia().diagonal_offsets(1));
  EXPECT_NE(Ah.dia().values(1, 0), Ahref_test.dia().values(1, 0));
  EXPECT_NE(Ah.csr().row_offsets(0), Ahref_test.csr().row_offsets(0));
  EXPECT_NE(Ah.csr().column_indices(0), Ahref_test.csr().column_indices(0));
  EXPECT_NE(Ah.csr().values(0), Ahref_test.csr().values(0));

  for (size_type i = this->Ahref.dia().diagonal_offsets().size();
       i < Ah.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Ah.dia().diagonal_offsets(i), 0);
  }

  for (size_type i = this->Ahref.dia().values().nrows();
       i < Ah.dia().values().nrows(); i++) {
    for (size_type j = this->Ahref.dia().values().ncols();
         j < Ah.dia().values().ncols(); j++) {
      EXPECT_EQ(Ah.dia().values(i, j), 0);
    }
  }

  for (size_type i = this->Ahref.csr().row_offsets().size();
       i < Ah.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Ah.csr().row_offsets(i), 0);
  }

  for (size_type i = this->Ahref.csr().values().size();
       i < Ah.csr().values().size(); i++) {
    EXPECT_EQ(Ah.csr().column_indices(i), 0);
    EXPECT_EQ(Ah.csr().values(i), 0);
  }

  // Resize to smaller shape and non-zeros
  size_type small_nrows = 2, small_ncols = 2, small_dia_nnnz = 2,
            small_csr_nnnz = 1, small_ndiag = 2;
  Matrix2 Asmall(small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                 small_ndiag);
  A.resize(Asmall);
  CHECK_HDC_CONTAINERS(Asmall, A);
  Ah.resize(Asmall);
  CHECK_HDC_CONTAINERS(Asmall, Ah);

  // Set back to normal
  Ah.dia().diagonal_offsets(1) = 0;
  Ah.dia().values(1, 0)        = (value_type)0;
  Ah.csr().row_offsets(0)      = 0;
  Ah.csr().column_indices(0)   = 7;
  Ah.csr().values(0)           = (value_type)3.33;
  Morpheus::copy(Ah, A);

  VALIDATE_HDC_CONTAINER(Ah, Ahref_test);
}

/**
 * @brief Testing allocation of HdcMatrix container from another HdcMatrix
 * container with the different parameters. New allocation shouldn't alias
 the
 * original.
 *
 */
TYPED_TEST(HdcMatrixBinaryTest, AllocateFromHdcMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using size_type   = typename Matrix1::size_type;
  using value_type1 = typename Matrix1::value_type;
  using value_type2 = typename Matrix2::value_type;

  HostMatrix1 Ah(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                 this->ndiag);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);
  Morpheus::Test::build_small_container(Ah);

  Matrix1 A(this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
            this->ndiag);
  CHECK_HDC_SIZES(A, this->nrows, this->ncols, this->dia_nnnz, this->csr_nnnz,
                  this->ndiag, this->nalign);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_HDC_EMPTY(Bh);

  Bh.allocate(Ah);
  CHECK_HDC_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type1)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type1)-1.11;

  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), 0);
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), 0);
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), 0);
    EXPECT_EQ(Bh.csr().values(i), 0);
  }

  // Now check device vector
  Matrix2 B;
  CHECK_HDC_EMPTY(B);

  Bh.dia().diagonal_offsets(1) = 2;
  Bh.dia().values(1, 0)        = (value_type2)-1.11;
  Bh.csr().row_offsets(0)      = 1;
  Bh.csr().column_indices(0)   = 1;
  Bh.csr().values(0)           = (value_type2)-1.11;

  B.allocate(A);
  CHECK_HDC_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (size_type i = 0; i < Bh.dia().diagonal_offsets().size(); i++) {
    EXPECT_EQ(Bh.dia().diagonal_offsets(i), 0);
  }

  for (size_type i = 0; i < Bh.dia().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.dia().values().ncols(); j++) {
      EXPECT_EQ(Bh.dia().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < Bh.csr().row_offsets().size(); i++) {
    EXPECT_EQ(Bh.csr().row_offsets(i), 0);
  }

  for (size_type i = 0; i < Bh.csr().values().size(); i++) {
    EXPECT_EQ(Bh.csr().column_indices(i), 0);
    EXPECT_EQ(Bh.csr().values(i), 0);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_HDCMATRIX_BINARY_HPP