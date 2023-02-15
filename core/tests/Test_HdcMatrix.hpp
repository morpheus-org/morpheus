/**
 * Test_HdcMatrix.hpp
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

#ifndef TEST_CORE_TEST_HDCMATRIX_HPP
#define TEST_CORE_TEST_HDCMATRIX_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HdcMatrix.hpp>

using HdcMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HdcMatrix<double>,
                                               types::types_set>::type;
using HdcMatrixUnary = to_gtest_types<HdcMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class HdcMatrixUnaryTest : public ::testing::Test {
 public:
  using type     = UnaryContainer;
  using device   = typename UnaryContainer::type;
  using host     = typename UnaryContainer::type::HostMirror;
  using SizeType = typename device::size_type;

  HdcMatrixUnaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        dia_nnz(SMALL_HDC_DIA_NNZ),
        csr_nnz(SMALL_HDC_CSR_NNZ),
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

  SizeType nrows, ncols, dia_nnz, csr_nnz, ndiag, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary HdcMatrix
 *
 */
TYPED_TEST_SUITE(HdcMatrixUnaryTest, HdcMatrixUnary);

/**
 * @brief Testing default construction of HdcMatrix container
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_HDC_EMPTY(A);
  EXPECT_EQ(A.dia().diagonal_offsets().data(), nullptr);
  EXPECT_EQ(A.dia().values().data(), nullptr);
  EXPECT_EQ(A.csr().row_offsets().data(), nullptr);
  EXPECT_EQ(A.csr().column_indices().data(), nullptr);
  EXPECT_EQ(A.csr().values().data(), nullptr);

  HostMatrix Ah;
  CHECK_HDC_EMPTY(Ah);
  EXPECT_EQ(Ah.dia().diagonal_offsets().data(), nullptr);
  EXPECT_EQ(Ah.dia().values().data(), nullptr);
  EXPECT_EQ(Ah.csr().row_offsets().data(), nullptr);
  EXPECT_EQ(Ah.csr().column_indices().data(), nullptr);
  EXPECT_EQ(Ah.csr().values().data(), nullptr);
}

/**
 * @brief Testing the enum value assigned to the container is what we expect
 it
 * to be i.e HDC_FORMAT.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(Morpheus::HDC_FORMAT, A.format_enum());
}

/**
 * @brief Testing the format index assigned to the container is what we
 expect
 * it to be i.e 3.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(5, A.format_index());
}

TYPED_TEST(HdcMatrixUnaryTest, ReferenceByIndex) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, this->nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);
  Morpheus::copy(A, Ah);

  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  Morpheus::Test::have_same_data(Ah.dia(), this->Ahref.dia());
  Morpheus::Test::have_same_data(Ah.csr(), this->Ahref.csr());
}

TYPED_TEST(HdcMatrixUnaryTest, Reference) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using dia_index_array_type =
      typename Matrix::dia_matrix_type::index_array_type;
  using dia_value_array_type =
      typename Matrix::dia_matrix_type::value_array_type;
  using dia_host_index_array_type = typename dia_index_array_type::HostMirror;
  using dia_host_value_array_type = typename dia_value_array_type::HostMirror;
  using csr_index_array_type =
      typename Matrix::csr_matrix_type::index_array_type;
  using csr_value_array_type =
      typename Matrix::csr_matrix_type::value_array_type;
  using csr_host_index_array_type = typename csr_index_array_type::HostMirror;
  using csr_host_value_array_type = typename csr_value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);
  Morpheus::copy(A, Ah);

  {
    dia_index_array_type doff = A.dia().diagonal_offsets();
    dia_value_array_type vals = A.dia().values();

    dia_host_index_array_type doff_h(A.dia().diagonal_offsets().size(), 0);
    Morpheus::copy(doff, doff_h);
    dia_host_value_array_type vals_h(A.dia().values().nrows(),
                                     A.dia().values().ncols(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type n = 0; n < A.dia().ndiags(); n++) {
      EXPECT_EQ(doff_h[n], this->Ahref.dia().diagonal_offsets(n));
    }

    for (size_type i = 0; i < A.dia().values().nrows(); i++) {
      for (size_type j = 0; j < A.dia().values().ncols(); j++) {
        EXPECT_EQ(vals_h(i, j), this->Ahref.dia().values(i, j));
      }
    }
  }

  {
    csr_index_array_type rind = A.csr().row_offsets();
    csr_index_array_type cind = A.csr().column_indices();
    csr_value_array_type vals = A.csr().values();

    csr_host_index_array_type rind_h(A.csr().row_offsets().size(), 0);
    Morpheus::copy(rind, rind_h);
    csr_host_index_array_type cind_h(A.csr().column_indices().size(), 0);
    Morpheus::copy(cind, cind_h);
    csr_host_value_array_type vals_h(A.csr().values().size(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.csr().row_offsets().size(); i++) {
      EXPECT_EQ(rind_h(i), this->Ahref.csr().row_offsets(i));
    }

    for (size_type i = 0; i < A.csr().values().size(); i++) {
      EXPECT_EQ(cind_h(i), this->Ahref.csr().column_indices(i));
      EXPECT_EQ(vals_h(i), this->Ahref.csr().values(i));
    }
  }
}

TYPED_TEST(HdcMatrixUnaryTest, ConstReference) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using dia_index_array_type =
      typename Matrix::dia_matrix_type::index_array_type;
  using dia_value_array_type =
      typename Matrix::dia_matrix_type::value_array_type;
  using dia_host_index_array_type = typename dia_index_array_type::HostMirror;
  using dia_host_value_array_type = typename dia_value_array_type::HostMirror;
  using csr_index_array_type =
      typename Matrix::csr_matrix_type::index_array_type;
  using csr_value_array_type =
      typename Matrix::csr_matrix_type::value_array_type;
  using csr_host_index_array_type = typename csr_index_array_type::HostMirror;
  using csr_host_value_array_type = typename csr_value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);
  Morpheus::copy(A, Ah);

  {
    dia_index_array_type doff = A.cdia().cdiagonal_offsets();
    dia_value_array_type vals = A.cdia().cvalues();

    dia_host_index_array_type doff_h(A.cdia().cdiagonal_offsets().size(), 0);
    Morpheus::copy(doff, doff_h);
    dia_host_value_array_type vals_h(A.cdia().cvalues().nrows(),
                                     A.cdia().cvalues().ncols(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type n = 0; n < A.cdia().ndiags(); n++) {
      EXPECT_EQ(doff_h[n], this->Ahref.cdia().cdiagonal_offsets(n));
    }

    for (size_type i = 0; i < A.cdia().cvalues().nrows(); i++) {
      for (size_type j = 0; j < A.cdia().cvalues().ncols(); j++) {
        EXPECT_EQ(vals_h(i, j), this->Ahref.cdia().cvalues(i, j));
      }
    }
  }

  {
    csr_index_array_type rind = A.ccsr().crow_offsets();
    csr_index_array_type cind = A.ccsr().ccolumn_indices();
    csr_value_array_type vals = A.ccsr().cvalues();

    csr_host_index_array_type rind_h(A.ccsr().crow_offsets().size(), 0);
    Morpheus::copy(rind, rind_h);
    csr_host_index_array_type cind_h(A.ccsr().ccolumn_indices().size(), 0);
    Morpheus::copy(cind, cind_h);
    csr_host_value_array_type vals_h(A.ccsr().cvalues().size(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.csr().row_offsets().size(); i++) {
      EXPECT_EQ(rind_h(i), this->Ahref.ccsr().crow_offsets(i));
    }

    for (size_type i = 0; i < A.csr().values().size(); i++) {
      EXPECT_EQ(cind_h(i), this->Ahref.ccsr().ccolumn_indices(i));
      EXPECT_EQ(vals_h(i), this->Ahref.ccsr().cvalues(i));
    }
  }
}

/**
 * @brief Testing default copy assignment of HdcMatrix container from another
 * HdcMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = Ah;
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-3.33;

  Ah.csr().row_offsets(0)    = 3;
  Ah.csr().column_indices(0) = 2;
  Ah.csr().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = A;
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default copy constructor of HdcMatrix container from
 another
 * HdcMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(Ah);
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-3.33;

  Ah.csr().row_offsets(0)    = 3;
  Ah.csr().column_indices(0) = 2;
  Ah.csr().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(A);
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move assignment of HdcMatrix container from another
 * HdcMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = std::move(Ah);
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-3.33;

  Ah.csr().row_offsets(0)    = 3;
  Ah.csr().column_indices(0) = 2;
  Ah.csr().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = std::move(A);
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move construction of HdcMatrix container from
 * another HdcMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(HdcMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.dia(), this->Aref.csr());
  CHECK_HDC_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.dia(), this->Ahref.csr());
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(std::move(Ah));
  CHECK_HDC_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-3.33;

  Ah.csr().row_offsets(0)    = 3;
  Ah.csr().column_indices(0) = 2;
  Ah.csr().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HDC_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(std::move(A));
  CHECK_HDC_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HDC_CONTAINER(Bt, Ah);
}

TYPED_TEST(HdcMatrixUnaryTest, ConstructionFromShapeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  HostMatrix Ah(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign);

  Morpheus::Test::build_small_container(Ah);

  Matrix A(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag);
  CHECK_HDC_SIZES(A, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, this->nalign)
  // Send to device
  Morpheus::copy(Ah, A);

  HostMatrix Ah_test(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                     this->ndiag);
  CHECK_HDC_SIZES(Ah_test, this->nrows, this->ncols, this->dia_nnz,
                  this->csr_nnz, this->ndiag, this->nalign)
  Morpheus::copy(A, Ah_test);

  VALIDATE_HDC_CONTAINER(Ah_test, Ah);
}

TYPED_TEST(HdcMatrixUnaryTest, ConstructionFromShape) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;

  HostMatrix Ah(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                this->ndiag, _nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign);
  Matrix A(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag,
           _nalign);
  CHECK_HDC_SIZES(A, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign)

  size_type _nalign1 = 512;

  HostMatrix Ah1(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                 this->ndiag, _nalign1);
  CHECK_HDC_SIZES(Ah1, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign1);
  Matrix A1(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag,
            _nalign1);
  CHECK_HDC_SIZES(A1, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign1)

  size_type _nalign2 = 333;

  HostMatrix Ah2(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                 this->ndiag, _nalign2);
  CHECK_HDC_SIZES(Ah2, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign2);
  Matrix A2(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag,
            _nalign2);
  CHECK_HDC_SIZES(A2, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign2)
}

// /**
//  * @brief Testing construction of HdcMatrix from a raw pointers
//  *
//  */
// TYPED_TEST(HdcMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1,0);
// }

TYPED_TEST(HdcMatrixUnaryTest, ResizeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using value_type = typename Matrix::value_type;

  size_type large_nrows = 500, large_ncols = 500, large_dia_nnnz = 640,
            large_csr_nnnz = 340, large_ndiag = 110;
  size_type small_nrows = 2, small_ncols = 2, small_dia_nnnz = 2,
            small_csr_nnnz = 1, small_ndiag = 2;

  Matrix A(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  A.resize(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
           large_ndiag);
  CHECK_HDC_SIZES(A, large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                  large_ndiag, this->nalign);

  HostMatrix Ah(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                large_ndiag);
  CHECK_HDC_SIZES(Ah, large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                  large_ndiag, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HDC_CONTAINER(this->Ahref, Ah);
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

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.dia().diagonal_offsets(1) = 2;
  Ah.dia().values(1, 0)        = (value_type)-1.11;
  Ah.csr().row_offsets(0)      = 1;
  Ah.csr().column_indices(0)   = 1;
  Ah.csr().values(0)           = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix Ahref_test(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                        this->ndiag);
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
  A.resize(small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
           small_ndiag);
  CHECK_HDC_SIZES(A, small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                  small_ndiag, this->nalign);
  Ah.resize(small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
            small_ndiag);
  CHECK_HDC_SIZES(Ah, small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                  small_ndiag, this->nalign);

  // Set back to normal
  Ah.dia().diagonal_offsets(1) = -3;
  Ah.dia().values(1, 0)        = (value_type)0;
  Ah.csr().row_offsets(0)      = 0;
  Ah.csr().column_indices(0)   = 7;
  Ah.csr().values(0)           = (value_type)3.33;
  Morpheus::copy(Ah, A);

  VALIDATE_HDC_CONTAINER(Ah, Ahref_test);
}

TYPED_TEST(HdcMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;
  HostMatrix Ah;
  CHECK_HDC_EMPTY(Ah);
  Ah.resize(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag,
            _nalign);
  CHECK_HDC_SIZES(Ah, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign);

  Matrix A;
  CHECK_HDC_EMPTY(A);
  A.resize(this->nrows, this->ncols, this->dia_nnz, this->csr_nnz, this->ndiag,
           _nalign);
  CHECK_HDC_SIZES(A, this->nrows, this->ncols, this->dia_nnz, this->csr_nnz,
                  this->ndiag, _nalign);

  size_type large_nrows = 500, large_ncols = 500, large_dia_nnnz = 640,
            large_csr_nnnz = 340, large_ndiag = 110, _nalign1 = 512;

  HostMatrix Ah1(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                 large_ndiag, _nalign1);
  CHECK_HDC_SIZES(Ah1, large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                  large_ndiag, _nalign1);

  Matrix A1(large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
            large_ndiag, _nalign1);
  CHECK_HDC_SIZES(A1, large_nrows, large_ncols, large_dia_nnnz, large_csr_nnnz,
                  large_ndiag, _nalign1);

  size_type small_nrows = 2, small_ncols = 2, small_dia_nnnz = 2,
            small_csr_nnnz = 1, small_ndiag = 2, _nalign2 = 333;

  HostMatrix Ah2(small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                 small_ndiag, _nalign2);
  CHECK_HDC_SIZES(Ah2, small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                  small_ndiag, _nalign2);

  Matrix A2(small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
            small_ndiag, _nalign2);
  CHECK_HDC_SIZES(A2, small_nrows, small_ncols, small_dia_nnnz, small_csr_nnnz,
                  small_ndiag, _nalign2)
}

TYPED_TEST(HdcMatrixUnaryTest, ResizeTolerance) {
  using Matrix    = typename TestFixture::device;
  using size_type = typename Matrix::size_type;

  Matrix A;
  CHECK_HDC_EMPTY(A);

  // Size above 100M entries
  A.resize(10e6, this->ncols, 60e6, 10, 15);
  CHECK_DIA_SIZES(A.dia(), size_type(10e6), this->ncols, size_type(60e6), 15,
                  this->nalign);

  // Fill ratio above 10
  A.resize(10, 10, 0, 0, 5);
  CHECK_HDC_SIZES(A, 10, 10, 0, 0, 5, this->nalign);
  A.resize(100, 100, 10, 2, 50);
  CHECK_HDC_SIZES(A, 100, 100, 10, 2, 50, this->nalign);

  // Both Size and Fill ratio above 100M and 10 respectively
  EXPECT_THROW(A.resize(size_type(10e6), this->ncols, 1000, 10, 15),
               Morpheus::FormatConversionException);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_HDCMATRIX_HPP