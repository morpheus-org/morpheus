/**
 * Test_EllMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_ELLMATRIX_BINARY_HPP
#define TEST_CORE_TEST_ELLMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_EllMatrix.hpp>

using EllMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::EllMatrix<double>,
                                               types::types_set>::type;
using EllMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        EllMatrixTypes, EllMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class EllMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // EllMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // EllMatrix
  using host2   = typename type2::type::HostMirror;

  using IndexType = typename device1::size_type;

  EllMatrixBinaryTest()
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

  IndexType nrows, ncols, nnnz, nentries_per_row, nalign;
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary EllMatrix pairs
 *
 */
TYPED_TEST_SUITE(EllMatrixBinaryTest, EllMatrixBinary);

TYPED_TEST(EllMatrixBinaryTest, ResizeFromEllMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using size_type   = typename Matrix1::size_type;
  using value_type  = typename Matrix1::value_type;

  Matrix1 A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  CHECK_ELL_SIZES(A, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  size_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
            large_nentries_per_row = 110;
  Matrix2 Alarge(large_nrows, large_ncols, large_nnnz, large_nentries_per_row);
  A.resize(Alarge);
  CHECK_ELL_CONTAINERS(Alarge, A);

  HostMatrix1 Ah(large_nrows, large_ncols, large_nnnz, large_nentries_per_row);
  CHECK_ELL_SIZES(Ah, large_nrows, large_ncols, large_nnnz,
                  large_nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);

  for (size_type i = 0; i < this->Ahref.values().nrows(); i++) {
    for (size_type j = 0; j < this->Ahref.values().ncols(); j++) {
      EXPECT_EQ(Ah.column_indices(i, j), this->Ahref.column_indices(i, j));
      EXPECT_EQ(Ah.values(i, j), this->Ahref.values(i, j));
    }
  }

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
  HostMatrix1 Ahref_test(this->nrows, this->ncols, this->nnnz,
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
  size_type small_nrows = 2, small_ncols = 2, small_nnnz = 2,
            small_nentries_per_row = 2;
  Matrix2 Asmall(small_nrows, small_ncols, small_nnnz, small_nentries_per_row);
  A.resize(Asmall);
  CHECK_ELL_CONTAINERS(Asmall, A);
  Ah.resize(Asmall);
  CHECK_ELL_CONTAINERS(Asmall, Ah);

  // Set back to normal
  Ah.column_indices(0, 1) = 3;
  Ah.values(0, 1)         = (value_type)2.22;
  Morpheus::copy(Ah, A);

  VALIDATE_ELL_CONTAINER(Ah, Ahref_test);
}

/**
 * @brief Testing allocation of EllMatrix container from another EllMatrix
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(EllMatrixBinaryTest, AllocateFromEllMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using size_type   = typename Matrix1::size_type;
  using value_type1 = typename Matrix1::value_type;
  using value_type2 = typename Matrix2::value_type;

  HostMatrix1 Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::Test::build_small_container(Ah);

  Matrix1 A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  CHECK_ELL_SIZES(A, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_ELL_EMPTY(Bh);

  Bh.allocate(Ah);
  CHECK_ELL_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.column_indices(0, 1) = 1;
  Ah.values(0, 1)         = (value_type1)-1.11;

  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), 0);
      EXPECT_EQ(Bh.values(i, j), 0);
    }
  }

  // Now check device vector
  Matrix2 B;
  CHECK_ELL_EMPTY(B);

  Bh.column_indices(0, 1) = 1;
  Bh.values(0, 1)         = (value_type2)-1.11;

  B.allocate(A);
  CHECK_ELL_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), 0);
      EXPECT_EQ(Bh.values(i, j), 0);
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_ELLMATRIX_BINARY_HPP