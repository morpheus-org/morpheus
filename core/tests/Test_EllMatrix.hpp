/**
 * Test_EllMatrix.hpp
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

#ifndef TEST_CORE_TEST_ELLMATRIX_HPP
#define TEST_CORE_TEST_ELLMATRIX_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_EllMatrix.hpp>

using EllMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::EllMatrix<double>,
                                               types::types_set>::type;
using EllMatrixUnary = to_gtest_types<EllMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class EllMatrixUnaryTest : public ::testing::Test {
 public:
  using type     = UnaryContainer;
  using device   = typename UnaryContainer::type;
  using host     = typename UnaryContainer::type::HostMirror;
  using SizeType = typename device::size_type;

  EllMatrixUnaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        nnnz(SMALL_MATRIX_NNZ),
        nentries_per_row(SMALL_ELL_ENTRIES_PER_ROW),
        nalign(SMALL_MATRIX_ALIGNMENT),
        Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
             SMALL_ELL_ENTRIES_PER_ROW, SMALL_MATRIX_ALIGNMENT),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
              SMALL_ELL_ENTRIES_PER_ROW, SMALL_MATRIX_ALIGNMENT) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  SizeType nrows, ncols, nnnz, nentries_per_row, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary EllMatrix
 *
 */
TYPED_TEST_SUITE(EllMatrixUnaryTest, EllMatrixUnary);

/**
 * @brief Testing default construction of EllMatrix container
 *
 */
TYPED_TEST(EllMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_ELL_EMPTY(A);
  EXPECT_EQ(A.column_indices().data(), nullptr);
  EXPECT_EQ(A.values().data(), nullptr);

  HostMatrix Ah;
  CHECK_ELL_EMPTY(Ah);
  EXPECT_EQ(Ah.column_indices().data(), nullptr);
  EXPECT_EQ(Ah.values().data(), nullptr);
}

/**
 * @brief Testing the enum value assigned to the container is what we expect
 it
 * to be i.e ELL_FORMAT.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(Morpheus::ELL_FORMAT, A.format_enum());
}

/**
 * @brief Testing the format index assigned to the container is what we
 expect
 * it to be i.e 3.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(3, A.format_index());
}

TYPED_TEST(EllMatrixUnaryTest, ReferenceByIndex) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  for (size_type i = 0; i < A.values().nrows(); i++) {
    for (size_type j = 0; j < A.values().ncols(); j++) {
      EXPECT_EQ(Ah.ccolumn_indices(i, j), this->Ahref.ccolumn_indices(i, j));
      EXPECT_EQ(Ah.cvalues(i, j), this->Ahref.values(i, j));
    }
  }
}

TYPED_TEST(EllMatrixUnaryTest, Reference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using size_type             = typename Matrix::size_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  index_array_type cind = A.column_indices();
  value_array_type vals = A.values();

  host_index_array_type cind_h(A.column_indices().nrows(),
                               A.column_indices().ncols(), 0);
  Morpheus::copy(cind, cind_h);
  host_value_array_type vals_h(A.values().nrows(), A.values().ncols(), 0);
  Morpheus::copy(vals, vals_h);

  for (size_type i = 0; i < A.values().nrows(); i++) {
    for (size_type j = 0; j < A.values().ncols(); j++) {
      EXPECT_EQ(cind_h(i, j), this->Ahref.column_indices(i, j));
      EXPECT_EQ(vals_h(i, j), this->Ahref.values(i, j));
    }
  }
}

TYPED_TEST(EllMatrixUnaryTest, ConstReference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using size_type             = typename Matrix::size_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  const index_array_type cind = A.ccolumn_indices();
  const value_array_type vals = A.cvalues();
  host_index_array_type cind_test(A.ccolumn_indices().nrows(),
                                  A.ccolumn_indices().ncols(), 0);
  Morpheus::copy(cind, cind_test);
  host_value_array_type vals_test(A.cvalues().nrows(), A.cvalues().ncols(), 0);
  Morpheus::copy(vals, vals_test);

  for (size_type i = 0; i < A.cvalues().nrows(); i++) {
    for (size_type j = 0; j < A.cvalues().ncols(); j++) {
      EXPECT_EQ(cind_test(i, j), this->Ahref.ccolumn_indices(i, j));
      EXPECT_EQ(vals_test(i, j), this->Ahref.values(i, j));
    }
  }

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah);

  const host_index_array_type cind_h = Ah.ccolumn_indices();
  const host_value_array_type vals_h = Ah.cvalues();

  for (size_type i = 0; i < Ah.values().nrows(); i++) {
    for (size_type j = 0; j < Ah.values().ncols(); j++) {
      EXPECT_EQ(cind_h(i, j), this->Ahref.column_indices(i, j));
      EXPECT_EQ(vals_h(i, j), this->Ahref.values(i, j));
    }
  }
}

/**
 * @brief Testing default copy assignment of EllMatrix container from another
 * EllMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = Ah;
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = A;
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default copy constructor of EllMatrix container from another
 * EllMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(Ah);
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(A);
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move assignment of EllMatrix container from another
 * EllMatrix container with the same parameters. Resulting container should
 be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh = std::move(Ah);
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B = std::move(A);
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing default move construction of EllMatrix container from
 * another EllMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(EllMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy asssignment
  HostMatrix Bh(std::move(Ah));
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix B(std::move(A));
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

TYPED_TEST(EllMatrixUnaryTest, ConstructionFromShapeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using value_type = typename Matrix::value_type;

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  for (size_type i = 0; i < Ah.values().nrows(); i++) {
    for (size_type j = 0; j < Ah.values().ncols(); j++) {
      EXPECT_EQ(Ah.column_indices(i, j), 0);
      EXPECT_EQ(Ah.values(i, j), (value_type)0);
    }
  }

  Morpheus::Test::build_small_container(Ah);

  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  CHECK_ELL_SIZES(A, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign)
  // Send to device
  Morpheus::copy(Ah, A);

  HostMatrix Ah_test(this->nrows, this->ncols, this->nnnz,
                     this->nentries_per_row);
  CHECK_ELL_SIZES(Ah_test, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign)
  Morpheus::copy(A, Ah_test);

  VALIDATE_ELL_CONTAINER(Ah_test, Ah);
}

TYPED_TEST(EllMatrixUnaryTest, ConstructionFromShape) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                _nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign);
  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           _nalign);
  CHECK_ELL_SIZES(A, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign)

  size_type _nalign1 = 512;

  HostMatrix Ah1(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 _nalign1);
  CHECK_ELL_SIZES(Ah1, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign1);
  Matrix A1(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
            _nalign1);
  CHECK_ELL_SIZES(A1, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign1)

  size_type _nalign2 = 333;

  HostMatrix Ah2(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 _nalign2);
  CHECK_ELL_SIZES(Ah2, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign2);
  Matrix A2(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
            _nalign2);
  CHECK_ELL_SIZES(A2, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign2)
}

// /**
//  * @brief Testing construction of EllMatrix from a raw pointers
//  *
//  */
// TYPED_TEST(EllMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1,0); }

TYPED_TEST(EllMatrixUnaryTest, ResizeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;
  using value_type = typename Matrix::value_type;

  size_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
            large_nentries_per_row = 110;
  size_type small_nrows = 2, small_ncols = 2, small_nnnz = 2,
            small_nentries_per_row = 2;

  Matrix A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  A.resize(large_nrows, large_ncols, large_nnnz, large_nentries_per_row);
  CHECK_ELL_SIZES(A, large_nrows, large_ncols, large_nnnz,
                  large_nentries_per_row, this->nalign);

  HostMatrix Ah(large_nrows, large_ncols, large_nnnz, large_nentries_per_row);
  CHECK_ELL_SIZES(Ah, large_nrows, large_ncols, large_nnnz,
                  large_nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(this->Ahref, Ah);
  for (size_type i = this->Ahref.values().nrows(); i < Ah.values().nrows();
       i++) {
    for (size_type j = this->Ahref.values().ncols(); j < Ah.values().ncols();
         j++) {
      EXPECT_EQ(Ah.column_indices(i, j), 0);
      EXPECT_EQ(Ah.values(i, j), 0);
    }
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.column_indices(0, 1) = 1;
  Ah.values(0, 1)         = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix Ahref_test(this->nrows, this->ncols, this->nnnz,
                        this->nentries_per_row);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.column_indices(0, 1), Ahref_test.column_indices(0, 1));
  EXPECT_NE(Ah.values(0, 1), Ahref_test.values(0, 1));

  for (size_type i = this->Ahref.values().nrows(); i < Ah.values().nrows();
       i++) {
    for (size_type j = this->Ahref.values().ncols(); j < Ah.values().ncols();
         j++) {
      EXPECT_EQ(Ah.column_indices(i, j), 0);
      EXPECT_EQ(Ah.values(i, j), 0);
    }
  }

  // Resize to smaller shape and non-zeros
  A.resize(small_nrows, small_ncols, small_nnnz, small_nentries_per_row);
  CHECK_ELL_SIZES(A, small_nrows, small_ncols, small_nnnz,
                  small_nentries_per_row, this->nalign);
  Ah.resize(small_nrows, small_ncols, small_nnnz, small_nentries_per_row);
  CHECK_ELL_SIZES(Ah, small_nrows, small_ncols, small_nnnz,
                  small_nentries_per_row, this->nalign);

  // Set back to normal
  Ah.column_indices(0, 1) = 3;
  Ah.values(0, 1)         = (value_type)2.22;
  Morpheus::copy(Ah, A);

  VALIDATE_ELL_CONTAINER(Ah, Ahref_test);
}

TYPED_TEST(EllMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using size_type  = typename Matrix::size_type;

  size_type _nalign = 127;
  HostMatrix Ah;
  CHECK_ELL_EMPTY(Ah);
  Ah.resize(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
            _nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign);

  Matrix A;
  CHECK_ELL_EMPTY(A);
  A.resize(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
           _nalign);
  CHECK_ELL_SIZES(A, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, _nalign);

  size_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
            large_nentries_per_row = 110, _nalign1 = 512;

  HostMatrix Ah1(large_nrows, large_ncols, large_nnnz, large_nentries_per_row,
                 _nalign1);
  CHECK_ELL_SIZES(Ah1, large_nrows, large_ncols, large_nnnz,
                  large_nentries_per_row, _nalign1);

  Matrix A1(large_nrows, large_ncols, large_nnnz, large_nentries_per_row,
            _nalign1);
  CHECK_ELL_SIZES(A1, large_nrows, large_ncols, large_nnnz,
                  large_nentries_per_row, _nalign1);

  size_type small_nrows = 2, small_ncols = 2, small_nnnz = 2,
            small_nentries_per_row = 2, _nalign2 = 333;

  HostMatrix Ah2(small_nrows, small_ncols, small_nnnz, small_nentries_per_row,
                 _nalign2);
  CHECK_ELL_SIZES(Ah2, small_nrows, small_ncols, small_nnnz,
                  small_nentries_per_row, _nalign2);

  Matrix A2(small_nrows, small_ncols, small_nnnz, small_nentries_per_row,
            _nalign2);
  CHECK_ELL_SIZES(A2, small_nrows, small_ncols, small_nnnz,
                  small_nentries_per_row, _nalign2)
}

// TYPED_TEST(EllMatrixUnaryTest, ResizeTolerance) {
//   using Matrix    = typename TestFixture::device;
//   using size_type = typename Matrix::size_type;

//   Matrix A;
//   CHECK_ELL_EMPTY(A);

//   // Size above 100M entries
//   A.resize(10e6, this->ncols, 60e6, 15);
//   CHECK_ELL_SIZES(A, size_type(10e6), this->ncols, size_type(60e6), 15,
//                   this->nalign);

//   // Fill ratio above 10
//   A.resize(10, 10, 0, 5);
//   CHECK_ELL_SIZES(A, 10, 10, 0, 5, this->nalign);
//   A.resize(100, 100, 10, 50);
//   CHECK_ELL_SIZES(A, 100, 100, 10, 50, this->nalign);

//   // Both Size and Fill ratio above 100M and 10 respectively
//   EXPECT_THROW(A.resize(size_type(10e6), this->ncols, 1000, 15),
//                Morpheus::FormatConversionException);
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_ELLMATRIX_HPP