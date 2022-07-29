/**
 * Test_DiaMatrix.hpp
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

#ifndef TEST_CORE_TEST_DIAMATRIX_HPP
#define TEST_CORE_TEST_DIAMATRIX_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DiaMatrix.hpp>

using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::types_set>::type;
using DiaMatrixUnary = to_gtest_types<DiaMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DiaMatrixUnaryTest : public ::testing::Test {
 public:
  using type      = UnaryContainer;
  using device    = typename UnaryContainer::type;
  using host      = typename UnaryContainer::type::HostMirror;
  using IndexType = typename device::index_type;

  DiaMatrixUnaryTest()
      : nrows(3),
        ncols(3),
        nnnz(4),
        ndiag(4),
        nalign(32),
        Aref(3, 3, 4, 4),
        Ahref(3, 3, 4, 4) {}

  void SetUp() override {
    build_diamatrix(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  IndexType nrows, ncols, nnnz, ndiag, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary DiaMatrix
 *
 */
TYPED_TEST_CASE(DiaMatrixUnaryTest, DiaMatrixUnary);

/**
 * @brief Testing default construction of DiaMatrix container
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_DIA_EMPTY(A);
  EXPECT_EQ(A.diagonal_offsets().data(), nullptr);
  EXPECT_EQ(A.values().data(), nullptr);

  HostMatrix Ah;
  CHECK_DIA_EMPTY(Ah);
  EXPECT_EQ(Ah.diagonal_offsets().data(), nullptr);
  EXPECT_EQ(Ah.values().data(), nullptr);
}

/**
 * @brief Testing the enum value assigned to the container is what we expect it
 * to be i.e DIA_FORMAT.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(Morpheus::DIA_FORMAT, A.format_enum());
}

/**
 * @brief Testing the format index assigned to the container is what we expect
 * it to be i.e 2.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(2, A.format_index());
}

TYPED_TEST(DiaMatrixUnaryTest, ReferenceByIndex) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  Morpheus::copy(A, Ah);

  VALIDATE_DIA_CONTAINER(Ah, this->Ahref, index_type);

  for (index_type n = 0; n < this->ndiag; n++) {
    EXPECT_EQ(Ah.cdiagonal_offsets(n), this->Ahref.diagonal_offsets(n));
  }

  for (index_type i = 0; i < A.values().nrows(); i++) {
    for (index_type j = 0; j < A.values().ncols(); j++) {
      EXPECT_EQ(Ah.cvalues(i, j), this->Ahref.values(i, j));
    }
  }
}

TYPED_TEST(DiaMatrixUnaryTest, Reference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  Morpheus::copy(A, Ah);

  index_array_type doff = A.diagonal_offsets();
  value_array_type vals = A.values();

  host_index_array_type doff_h(A.ndiags(), 0);
  Morpheus::copy(doff, doff_h);
  host_value_array_type vals_h(A.values().nrows(), A.values().ncols(), 0);
  Morpheus::copy(vals, vals_h);

  for (index_type n = 0; n < A.ndiags(); n++) {
    EXPECT_EQ(doff_h[n], this->Ahref.diagonal_offsets(n));
  }

  for (index_type i = 0; i < A.values().nrows(); i++) {
    for (index_type j = 0; j < A.values().ncols(); j++) {
      EXPECT_EQ(vals_h(i, j), this->Ahref.values(i, j));
    }
  }
}

TYPED_TEST(DiaMatrixUnaryTest, ConstReference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  const index_array_type doff = A.cdiagonal_offsets();
  const value_array_type vals = A.cvalues();
  host_index_array_type doff_test(A.ndiags(), 0);
  Morpheus::copy(doff, doff_test);
  host_value_array_type vals_test(A.values().nrows(), A.values().ncols(), 0);
  Morpheus::copy(vals, vals_test);

  for (index_type n = 0; n < A.ndiags(); n++) {
    EXPECT_EQ(doff_test[n], this->Ahref.diagonal_offsets(n));
  }
  for (index_type i = 0; i < A.values().nrows(); i++) {
    for (index_type j = 0; j < A.values().ncols(); j++) {
      EXPECT_EQ(vals_test(i, j), this->Ahref.values(i, j));
    }
  }

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  Morpheus::copy(A, Ah);

  const host_index_array_type doff_h = Ah.cdiagonal_offsets();
  const host_value_array_type vals_h = Ah.cvalues();

  for (index_type n = 0; n < Ah.ndiags(); n++) {
    EXPECT_EQ(doff_h[n], this->Ahref.diagonal_offsets(n));
  }
  for (index_type i = 0; i < Ah.values().nrows(); i++) {
    for (index_type j = 0; j < Ah.values().ncols(); j++) {
      EXPECT_EQ(vals_h(i, j), this->Ahref.values(i, j));
    }
  }
}

/**
 * @brief Testing default copy assignment of DiaMatrix container from another
 * DiaMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref, index_type);

  // Default copy asssignment
  HostMatrix Bh = Ah;
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(2) = 2;
  Ah.values(0, 1)        = -3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah, index_type);

  // Now check device Matrix
  Matrix B = A;
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah, index_type);
}

/**
 * @brief Testing default copy constructor of DiaMatrix container from another
 * DiaMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref, index_type);

  // Default copy asssignment
  HostMatrix Bh(Ah);
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(2) = 2;
  Ah.values(0, 1)        = -3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah, index_type);

  // Now check device Matrix
  Matrix B(A);
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah, index_type);
}

/**
 * @brief Testing default move assignment of DiaMatrix container from another
 * DiaMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref, index_type);

  // Default copy asssignment
  HostMatrix Bh = std::move(Ah);
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(2) = 2;
  Ah.values(0, 1)        = -3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah, index_type);

  // Now check device Matrix
  Matrix B = std::move(A);
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah, index_type);
}

/**
 * @brief Testing default move construction of DiaMatrix container from
 * another DiaMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DiaMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref, index_type);

  // Default copy asssignment
  HostMatrix Bh(std::move(Ah));
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(2) = 2;
  Ah.values(0, 1)        = -3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah, index_type);

  // Now check device Matrix
  Matrix B(std::move(A));
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah, index_type);
}

TYPED_TEST(DiaMatrixUnaryTest, ConstructionFromShapeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  for (index_type n = 0; n < Ah.ndiags(); n++) {
    EXPECT_EQ(Ah.diagonal_offsets(n), (index_type)0);
  }
  for (index_type i = 0; i < Ah.values().nrows(); i++) {
    for (index_type j = 0; j < Ah.values().ncols(); j++) {
      EXPECT_EQ(Ah.values(i, j), (value_type)0);
    }
  }

  build_diamatrix(Ah);

  Matrix A(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(A, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign)
  // Send to device
  Morpheus::copy(Ah, A);

  HostMatrix Ah_test(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(Ah_test, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign)
  Morpheus::copy(A, Ah_test);

  VALIDATE_DIA_CONTAINER(Ah_test, Ah, index_type);
}

TYPED_TEST(DiaMatrixUnaryTest, ConstructionFromShape) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  index_type _nalign = 127;

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign);
  Matrix A(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign);
  CHECK_DIA_SIZES(A, this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign)

  index_type _nalign1 = 512;

  HostMatrix Ah1(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign1);
  CHECK_DIA_SIZES(Ah1, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign1);
  Matrix A1(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign1);
  CHECK_DIA_SIZES(A1, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign1)

  index_type _nalign2 = 333;

  HostMatrix Ah2(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign2);
  CHECK_DIA_SIZES(Ah2, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign2);
  Matrix A2(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign2);
  CHECK_DIA_SIZES(A2, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign2)
}

// /**
//  * @brief Testing construction of DiaMatrix from a raw pointers
//  *
//  */
// TYPED_TEST(DiaMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1, 0); }

TYPED_TEST(DiaMatrixUnaryTest, ResizeDefault) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  index_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
             large_ndiag = 110;
  index_type small_nrows = 2, small_ncols = 2, small_nnnz = 2, small_ndiag = 2;

  Matrix A(this->nrows, this->ncols, this->nnnz, this->ndiag);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  A.resize(large_nrows, large_ncols, large_nnnz, large_ndiag);
  CHECK_DIA_SIZES(A, large_nrows, large_ncols, large_nnnz, large_ndiag,
                  this->nalign);

  HostMatrix Ah(large_nrows, large_ncols, large_nnnz, large_ndiag);
  CHECK_DIA_SIZES(Ah, large_nrows, large_ncols, large_nnnz, large_ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(this->Ahref, Ah, index_type);
  for (index_type n = this->ndiag; n < Ah.ndiags(); n++) {
    EXPECT_EQ(Ah.diagonal_offsets(n), 0);
  }
  for (index_type i = this->Ahref.values().nrows(); i < Ah.values().nrows();
       i++) {
    for (index_type j = this->Ahref.values().ncols(); j < Ah.values().ncols();
         j++) {
      EXPECT_EQ(Ah.values(i, j), 0);
    }
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.diagonal_offsets(1) = 1;
  Ah.values(0, 1)        = -1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix Ahref_test(this->nrows, this->ncols, this->nnnz, this->ndiag);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.diagonal_offsets(1), Ahref_test.diagonal_offsets(1));
  EXPECT_NE(Ah.values(0, 1), Ahref_test.values(0, 1));

  for (index_type n = this->ndiag; n < Ah.ndiags(); n++) {
    EXPECT_EQ(Ah.diagonal_offsets(n), 0);
  }
  for (index_type i = this->Ahref.values().nrows(); i < Ah.values().nrows();
       i++) {
    for (index_type j = this->Ahref.values().ncols(); j < Ah.values().ncols();
         j++) {
      EXPECT_EQ(Ah.values(i, j), 0);
    }
  }

  // Resize to smaller shape and non-zeros
  A.resize(small_nrows, small_ncols, small_nnnz, small_ndiag);
  CHECK_DIA_SIZES(A, small_nrows, small_ncols, small_nnnz, small_ndiag,
                  this->nalign);
  Ah.resize(small_nrows, small_ncols, small_nnnz, small_ndiag);
  CHECK_DIA_SIZES(Ah, small_nrows, small_ncols, small_nnnz, small_ndiag,
                  this->nalign);

  // Set back to normal
  Ah.diagonal_offsets(1) = 0;
  Ah.values(0, 1)        = 1.11;
  Morpheus::copy(Ah, A);

  VALIDATE_DIA_CONTAINER(Ah, Ahref_test, index_type);
}

TYPED_TEST(DiaMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  index_type _nalign = 127;
  HostMatrix Ah;
  CHECK_DIA_EMPTY(Ah);
  Ah.resize(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign);

  Matrix A;
  CHECK_DIA_EMPTY(A);
  A.resize(this->nrows, this->ncols, this->nnnz, this->ndiag, _nalign);
  CHECK_DIA_SIZES(A, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  _nalign);

  index_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
             large_ndiag = 110, _nalign1 = 512;

  HostMatrix Ah1(large_nrows, large_ncols, large_nnnz, large_ndiag, _nalign1);
  CHECK_DIA_SIZES(Ah1, large_nrows, large_ncols, large_nnnz, large_ndiag,
                  _nalign1);

  Matrix A1(large_nrows, large_ncols, large_nnnz, large_ndiag, _nalign1);
  CHECK_DIA_SIZES(A1, large_nrows, large_ncols, large_nnnz, large_ndiag,
                  _nalign1);

  index_type small_nrows = 2, small_ncols = 2, small_nnnz = 2, small_ndiag = 2,
             _nalign2 = 333;

  HostMatrix Ah2(small_nrows, small_ncols, small_nnnz, small_ndiag, _nalign2);
  CHECK_DIA_SIZES(Ah2, small_nrows, small_ncols, small_nnnz, small_ndiag,
                  _nalign2);

  Matrix A2(small_nrows, small_ncols, small_nnnz, small_ndiag, _nalign2);
  CHECK_DIA_SIZES(A2, small_nrows, small_ncols, small_nnnz, small_ndiag,
                  _nalign2)
}

TYPED_TEST(DiaMatrixUnaryTest, ResizeTolerance) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;

  Matrix A;
  CHECK_DIA_EMPTY(A);

  // Size above 100M entries
  A.resize(10e6, this->ncols, 60e6, 15);
  CHECK_DIA_SIZES(A, index_type(10e6), this->ncols, index_type(60e6), 15,
                  this->nalign);

  // Fill ratio above 10
  A.resize(10, 10, 0, 5);
  CHECK_DIA_SIZES(A, 10, 10, 0, 5, this->nalign);
  A.resize(100, 100, 10, 50);
  CHECK_DIA_SIZES(A, 100, 100, 10, 50, this->nalign);

  // Both Size and Fill ratio above 100M and 10 respectively
  EXPECT_THROW(A.resize(10e6, this->ncols, 1000, 15),
               Morpheus::FormatConversionException);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DIAMATRIX_HPP