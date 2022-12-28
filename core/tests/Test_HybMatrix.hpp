/**
 * Test_HybMatrix.hpp
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

#ifndef TEST_CORE_TEST_HYBMATRIX_HPP
#define TEST_CORE_TEST_HYBMATRIX_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HybMatrix.hpp>

using HybMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HybMatrix<double>,
                                               types::types_set>::type;
using HybMatrixUnary = to_gtest_types<HybMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class HybMatrixUnaryTest : public ::testing::Test {
 public:
  using type     = UnaryContainer;
  using device   = typename UnaryContainer::type;
  using host     = typename UnaryContainer::type::HostMirror;
  using SizeType = typename device::size_type;

  HybMatrixUnaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        ell_nnz(SMALL_HYB_ELL_NNZ),
        coo_nnz(SMALL_HYB_COO_NNZ),
        nentries_per_row(SMALL_HYB_ENTRIES_PER_ROW),
        nalign(SMALL_MATRIX_ALIGNMENT),
        Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HYB_ELL_NNZ,
             SMALL_HYB_COO_NNZ, SMALL_HYB_ENTRIES_PER_ROW,
             SMALL_MATRIX_ALIGNMENT),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_HYB_ELL_NNZ,
              SMALL_HYB_COO_NNZ, SMALL_HYB_ENTRIES_PER_ROW,
              SMALL_MATRIX_ALIGNMENT) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  SizeType nrows, ncols, ell_nnz, coo_nnz, nentries_per_row, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary HybMatrix
 *
 */
TYPED_TEST_SUITE(HybMatrixUnaryTest, HybMatrixUnary);

/**
 * @brief Testing default construction of HybMatrix container
 *
 */
TYPED_TEST(HybMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_HYB_EMPTY(A);
  EXPECT_EQ(A.ell().column_indices().data(), nullptr);
  EXPECT_EQ(A.ell().values().data(), nullptr);
  EXPECT_EQ(A.coo().row_indices().data(), nullptr);
  EXPECT_EQ(A.coo().column_indices().data(), nullptr);
  EXPECT_EQ(A.coo().values().data(), nullptr);

  HostMatrix Ah;
  CHECK_HYB_EMPTY(Ah);
  EXPECT_EQ(Ah.ell().column_indices().data(), nullptr);
  EXPECT_EQ(Ah.ell().values().data(), nullptr);
  EXPECT_EQ(Ah.coo().row_indices().data(), nullptr);
  EXPECT_EQ(Ah.coo().column_indices().data(), nullptr);
  EXPECT_EQ(Ah.coo().values().data(), nullptr);
}

/**
 * @brief Testing the enum value assigned to the container is what we expect
 it
 * to be i.e HYB_FORMAT.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(Morpheus::HYB_FORMAT, A.format_enum());
}

/**
 * @brief Testing the format index assigned to the container is what we
 expect
 * it to be i.e 3.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(4, A.format_index());
}

TYPED_TEST(HybMatrixUnaryTest, ReferenceByIndex) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, this->nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  Morpheus::Test::have_same_data(Ah.ell(), this->Ahref.ell());
  Morpheus::Test::have_same_data(Ah.coo(), this->Ahref.coo());
}

TYPED_TEST(HybMatrixUnaryTest, Reference) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using ell_index_array_type =
      typename Matrix::ell_matrix_type::index_array_type;
  using ell_value_array_type =
      typename Matrix::ell_matrix_type::value_array_type;
  using ell_host_index_array_type = typename ell_index_array_type::HostMirror;
  using ell_host_value_array_type = typename ell_value_array_type::HostMirror;
  using coo_index_array_type =
      typename Matrix::coo_matrix_type::index_array_type;
  using coo_value_array_type =
      typename Matrix::coo_matrix_type::value_array_type;
  using coo_host_index_array_type = typename coo_index_array_type::HostMirror;
  using coo_host_value_array_type = typename coo_value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  {
    ell_index_array_type cind = A.ell().column_indices();
    ell_value_array_type vals = A.ell().values();

    ell_host_index_array_type cind_h(A.ell().column_indices().nrows(),
                                     A.ell().column_indices().ncols(), 0);
    Morpheus::copy(cind, cind_h);
    ell_host_value_array_type vals_h(A.ell().values().nrows(),
                                     A.ell().values().ncols(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.ell().values().nrows(); i++) {
      for (size_type j = 0; j < A.ell().values().ncols(); j++) {
        EXPECT_EQ(cind_h(i, j), this->Ahref.ell().column_indices(i, j));
        EXPECT_EQ(vals_h(i, j), this->Ahref.ell().values(i, j));
      }
    }
  }

  {
    coo_index_array_type rind = A.coo().row_indices();
    coo_index_array_type cind = A.coo().column_indices();
    coo_value_array_type vals = A.coo().values();

    coo_host_index_array_type rind_h(A.coo().row_indices().size(), 0);
    Morpheus::copy(rind, rind_h);
    coo_host_index_array_type cind_h(A.coo().column_indices().size(), 0);
    Morpheus::copy(cind, cind_h);
    coo_host_value_array_type vals_h(A.coo().values().size(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.coo().values().size(); i++) {
      EXPECT_EQ(rind_h(i), this->Ahref.coo().row_indices(i));
      EXPECT_EQ(cind_h(i), this->Ahref.coo().column_indices(i));
      EXPECT_EQ(vals_h(i), this->Ahref.coo().values(i));
    }
  }
}

TYPED_TEST(HybMatrixUnaryTest, ConstReference) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using ell_index_array_type =
      typename Matrix::ell_matrix_type::index_array_type;
  using ell_value_array_type =
      typename Matrix::ell_matrix_type::value_array_type;
  using ell_host_index_array_type = typename ell_index_array_type::HostMirror;
  using ell_host_value_array_type = typename ell_value_array_type::HostMirror;
  using coo_index_array_type =
      typename Matrix::coo_matrix_type::index_array_type;
  using coo_value_array_type =
      typename Matrix::coo_matrix_type::value_array_type;
  using coo_host_index_array_type = typename coo_index_array_type::HostMirror;
  using coo_host_value_array_type = typename coo_value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  {
    ell_index_array_type cind = A.cell().ccolumn_indices();
    ell_value_array_type vals = A.cell().cvalues();

    ell_host_index_array_type cind_h(A.cell().ccolumn_indices().nrows(),
                                     A.cell().ccolumn_indices().ncols(), 0);
    Morpheus::copy(cind, cind_h);
    ell_host_value_array_type vals_h(A.cell().cvalues().nrows(),
                                     A.cell().cvalues().ncols(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.cell().cvalues().nrows(); i++) {
      for (size_type j = 0; j < A.cell().cvalues().ncols(); j++) {
        EXPECT_EQ(cind_h(i, j), this->Ahref.cell().ccolumn_indices(i, j));
        EXPECT_EQ(vals_h(i, j), this->Ahref.cell().cvalues(i, j));
      }
    }
  }

  {
    coo_index_array_type rind = A.ccoo().crow_indices();
    coo_index_array_type cind = A.ccoo().ccolumn_indices();
    coo_value_array_type vals = A.ccoo().cvalues();

    coo_host_index_array_type rind_h(A.ccoo().crow_indices().size(), 0);
    Morpheus::copy(rind, rind_h);
    coo_host_index_array_type cind_h(A.ccoo().ccolumn_indices().size(), 0);
    Morpheus::copy(cind, cind_h);
    coo_host_value_array_type vals_h(A.ccoo().cvalues().size(), 0);
    Morpheus::copy(vals, vals_h);

    for (size_type i = 0; i < A.coo().values().size(); i++) {
      EXPECT_EQ(rind_h(i), this->Ahref.ccoo().crow_indices(i));
      EXPECT_EQ(cind_h(i), this->Ahref.ccoo().ccolumn_indices(i));
      EXPECT_EQ(vals_h(i), this->Ahref.ccoo().cvalues(i));
    }
  }
}

/**
 * @brief Testing default copy assignment of HybMatrix container from another
 * HybMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = Ah;
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 3;
  Ah.ell().values(1, 0)         = (value_type)-3.33;

  Ah.coo().row_indices(0)    = 3;
  Ah.coo().column_indices(0) = 2;
  Ah.coo().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = A;
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default copy constructor of HybMatrix container from
 another
 * HybMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(Ah);
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 3;
  Ah.ell().values(1, 0)         = (value_type)-3.33;

  Ah.coo().row_indices(0)    = 3;
  Ah.coo().column_indices(0) = 2;
  Ah.coo().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(A);
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move assignment of HybMatrix container from another
 * HybMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = std::move(Ah);
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 3;
  Ah.ell().values(1, 0)         = (value_type)-3.33;

  Ah.coo().row_indices(0)    = 3;
  Ah.coo().column_indices(0) = 2;
  Ah.coo().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = std::move(A);
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move construction of HybMatrix container from
 * another HybMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(HybMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->Ahref.ell(), this->Ahref.coo());
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(std::move(Ah));
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 3;
  Ah.ell().values(1, 0)         = (value_type)-3.33;

  Ah.coo().row_indices(0)    = 3;
  Ah.coo().column_indices(0) = 2;
  Ah.coo().values(0)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(std::move(A));
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

TYPED_TEST(HybMatrixUnaryTest, ConstructionFromShapeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  HostMatrix Ah(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::Test::build_small_container(Ah);

  Matrix A(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
           this->nentries_per_row);
  CHECK_HYB_SIZES(A, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, this->nalign)
  // Send to device
  Morpheus::copy(Ah, A);

  HostMatrix Ah_test(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                     this->nentries_per_row);
  CHECK_HYB_SIZES(Ah_test, this->nrows, this->ncols, this->ell_nnz,
                  this->coo_nnz, this->nentries_per_row, this->nalign)
  Morpheus::copy(A, Ah_test);

  VALIDATE_HYB_CONTAINER(Ah_test, Ah);
}

TYPED_TEST(HybMatrixUnaryTest, ConstructionFromShape) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;

  HostMatrix Ah(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                this->nentries_per_row, _nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign);
  Matrix A(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
           this->nentries_per_row, _nalign);
  CHECK_HYB_SIZES(A, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign)

  size_type _nalign1 = 512;

  HostMatrix Ah1(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                 this->nentries_per_row, _nalign1);
  CHECK_HYB_SIZES(Ah1, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign1);
  Matrix A1(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
            this->nentries_per_row, _nalign1);
  CHECK_HYB_SIZES(A1, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign1)

  size_type _nalign2 = 333;

  HostMatrix Ah2(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                 this->nentries_per_row, _nalign2);
  CHECK_HYB_SIZES(Ah2, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign2);
  Matrix A2(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
            this->nentries_per_row, _nalign2);
  CHECK_HYB_SIZES(A2, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign2)
}

// /**
//  * @brief Testing construction of HybMatrix from a raw pointers
//  *
//  */
// TYPED_TEST(HybMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1,0);
// }

TYPED_TEST(HybMatrixUnaryTest, ResizeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using value_type = typename Matrix::value_type;

  size_type large_nrows = 500, large_ncols = 500, large_ell_nnnz = 640,
            large_coo_nnnz = 340, large_nentries_per_row = 110;
  size_type small_nrows = 2, small_ncols = 2, small_ell_nnnz = 2,
            small_coo_nnnz = 1, small_nentries_per_row = 2;

  Matrix A(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
           this->nentries_per_row);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  A.resize(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
           large_nentries_per_row);
  CHECK_HYB_SIZES(A, large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                  large_nentries_per_row, this->nalign);

  HostMatrix Ah(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                large_nentries_per_row);
  CHECK_HYB_SIZES(Ah, large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                  large_nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(this->Ahref, Ah);
  for (size_type i = this->Ahref.ell().values().nrows();
       i < Ah.ell().values().nrows(); i++) {
    for (size_type j = this->Ahref.ell().values().ncols();
         j < Ah.ell().values().ncols(); j++) {
      EXPECT_EQ(Ah.ell().column_indices(i, j), 0);
      EXPECT_EQ(Ah.ell().values(i, j), 0);
    }
  }

  for (size_type i = this->Ahref.coo().values().size();
       i < Ah.coo().values().size(); i++) {
    EXPECT_EQ(Ah.coo().row_indices(i), 0);
    EXPECT_EQ(Ah.coo().column_indices(i), 0);
    EXPECT_EQ(Ah.coo().values(i), 0);
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.ell().column_indices(1, 0) = 1;
  Ah.ell().values(1, 0)         = (value_type)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix Ahref_test(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                        this->nentries_per_row);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.ell().column_indices(1, 0),
            Ahref_test.ell().column_indices(1, 0));
  EXPECT_NE(Ah.ell().values(1, 0), Ahref_test.ell().values(1, 0));
  EXPECT_NE(Ah.coo().row_indices(0), Ahref_test.coo().row_indices(0));
  EXPECT_NE(Ah.coo().column_indices(0), Ahref_test.coo().column_indices(0));
  EXPECT_NE(Ah.coo().values(0), Ahref_test.coo().values(0));

  for (size_type i = this->Ahref.ell().values().nrows();
       i < Ah.ell().values().nrows(); i++) {
    for (size_type j = this->Ahref.ell().values().ncols();
         j < Ah.ell().values().ncols(); j++) {
      EXPECT_EQ(Ah.ell().column_indices(i, j), 0);
      EXPECT_EQ(Ah.ell().values(i, j), 0);
    }
  }

  for (size_type i = this->Ahref.coo().values().size();
       i < Ah.coo().values().size(); i++) {
    EXPECT_EQ(Ah.coo().row_indices(i), 0);
    EXPECT_EQ(Ah.coo().column_indices(i), 0);
    EXPECT_EQ(Ah.coo().values(i), 0);
  }

  // Resize to smaller shape and non-zeros
  A.resize(small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
           small_nentries_per_row);
  CHECK_HYB_SIZES(A, small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                  small_nentries_per_row, this->nalign);
  Ah.resize(small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
            small_nentries_per_row);
  CHECK_HYB_SIZES(Ah, small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                  small_nentries_per_row, this->nalign);

  // Set back to normal
  Ah.ell().column_indices(1, 0) = 0;
  Ah.ell().values(1, 0)         = (value_type)5.55;
  Ah.coo().row_indices(0)       = 0;
  Ah.coo().column_indices(0)    = 3;
  Ah.coo().values(0)            = (value_type)4.44;
  Morpheus::copy(Ah, A);

  VALIDATE_HYB_CONTAINER(Ah, Ahref_test);
}

TYPED_TEST(HybMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;
  HostMatrix Ah;
  CHECK_HYB_EMPTY(Ah);
  Ah.resize(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
            this->nentries_per_row, _nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign);

  Matrix A;
  CHECK_HYB_EMPTY(A);
  A.resize(this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
           this->nentries_per_row, _nalign);
  CHECK_HYB_SIZES(A, this->nrows, this->ncols, this->ell_nnz, this->coo_nnz,
                  this->nentries_per_row, _nalign);

  size_type large_nrows = 500, large_ncols = 500, large_ell_nnnz = 640,
            large_coo_nnnz = 340, large_nentries_per_row = 110, _nalign1 = 512;

  HostMatrix Ah1(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                 large_nentries_per_row, _nalign1);
  CHECK_HYB_SIZES(Ah1, large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                  large_nentries_per_row, _nalign1);

  Matrix A1(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
            large_nentries_per_row, _nalign1);
  CHECK_HYB_SIZES(A1, large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                  large_nentries_per_row, _nalign1);

  size_type small_nrows = 2, small_ncols = 2, small_ell_nnnz = 2,
            small_coo_nnnz = 1, small_nentries_per_row = 2, _nalign2 = 333;

  HostMatrix Ah2(small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                 small_nentries_per_row, _nalign2);
  CHECK_HYB_SIZES(Ah2, small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                  small_nentries_per_row, _nalign2);

  Matrix A2(small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
            small_nentries_per_row, _nalign2);
  CHECK_HYB_SIZES(A2, small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                  small_nentries_per_row, _nalign2)
}

// // TYPED_TEST(HybMatrixUnaryTest, ResizeTolerance) {
// //   using Matrix    = typename TestFixture::device;
// //   using size_type = typename Matrix::size_type;

// //   Matrix A;
// //   CHECK_HYB_EMPTY(A);

// //   // Size above 100M entries
// //   A.resize(10e6, this->ncols, 60e6, 15);
// //   CHECK_HYB_SIZES(A, size_type(10e6), this->ncols, size_type(60e6), 15,
// //                   this->nalign);

// //   // Fill ratio above 10
// //   A.resize(10, 10, 0, 5);
// //   CHECK_HYB_SIZES(A, 10, 10, 0, 5, this->nalign);
// //   A.resize(100, 100, 10, 50);
// //   CHECK_HYB_SIZES(A, 100, 100, 10, 50, this->nalign);

// //   // Both Size and Fill ratio above 100M and 10 respectively
// //   EXPECT_THROW(A.resize(size_type(10e6), this->ncols, 1000, 15),
// //                Morpheus::FormatConversionException);
// // }

}  // namespace Test

#endif  // TEST_CORE_TEST_HYBMATRIX_HPP