/**
 * Test_CsrMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_CSRMATRIX_BINARY_HPP
#define TEST_CORE_TEST_CSRMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_CsrMatrix.hpp>

using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types::types_set>::type;
using CsrMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CsrMatrixTypes, CsrMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class CsrMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CsrMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CsrMatrix
  using host2   = typename type2::type::HostMirror;

  CsrMatrixBinaryTest()
      : Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary CsrMatrix pairs
 *
 */
TYPED_TEST_SUITE(CsrMatrixBinaryTest, CsrMatrixBinary);

TYPED_TEST(CsrMatrixBinaryTest, ResizeFromCsrMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using size_type   = typename Matrix1::size_type;
  using value_type  = typename Matrix1::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz        = SMALL_MATRIX_NNZ;
  size_type large_nrows = 500, large_ncols = 400, large_nnnz = 640;
  size_type small_nrows = 2, small_ncols = 3, small_nnnz = 4;

  Matrix1 A(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(A, nrows, ncols, nnnz);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  Matrix2 Alarge(large_nrows, large_ncols, large_nnnz);
  A.resize(Alarge);
  CHECK_CSR_CONTAINERS(Alarge, A);

  HostMatrix1 Ah(large_nrows, large_ncols, large_nnnz);
  CHECK_CSR_SIZES(Ah, large_nrows, large_ncols, large_nnnz);
  Morpheus::copy(A, Ah);

  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Ah.row_offsets(n), this->Ahref.row_offsets(n));
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.column_indices(n), this->Ahref.column_indices(n));
    EXPECT_EQ(Ah.values(n), this->Ahref.values(n));
  }

  for (size_type n = nrows + 1; n < Ah.nrows() + 1; n++) {
    EXPECT_EQ(Ah.row_offsets(n), 0);
  }
  for (size_type n = nnnz; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.column_indices(n), 0);
    EXPECT_EQ(Ah.values(n), 0);
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.row_offsets(2)    = 6;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = (value_type)-3.33;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix1 Ahref_test(nrows, ncols, nnnz);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.row_offsets(2), Ahref_test.row_offsets(2));
  EXPECT_NE(Ah.column_indices(1), Ahref_test.column_indices(1));
  EXPECT_NE(Ah.values(3), Ahref_test.values(3));

  for (size_type n = nrows + 1; n < Ah.nrows() + 1; n++) {
    EXPECT_EQ(Ah.row_offsets(n), 0);
  }
  for (size_type n = nnnz; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.column_indices(n), 0);
    EXPECT_EQ(Ah.values(n), (value_type)0);
  }
  Matrix2 Asmall(small_nrows, small_ncols, small_nnnz);
  // Resize to smaller shape and non-zeros
  A.resize(Asmall);
  CHECK_CSR_CONTAINERS(Asmall, A);
  Ah.resize(Asmall);
  CHECK_CSR_CONTAINERS(Asmall, Ah);

  // Set back to normal
  Ah.row_offsets(2)    = 8;
  Ah.column_indices(1) = 3;
  Ah.values(3)         = (value_type)4.44;

  Morpheus::copy(Ah, A);
  for (size_type n = 0; n < Ah.nrows() + 1; n++) {
    EXPECT_EQ(Ah.row_offsets(n), Ahref_test.row_offsets(n));
  }
  for (size_type n = 0; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.column_indices(n), Ahref_test.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref_test.values(n));
  }
}

/**
 * @brief Testing allocation of CsrMatrix container from another CsrMatrix
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(CsrMatrixBinaryTest, AllocateFromCsrMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using size_type   = typename Matrix1::size_type;
  using value_type1 = typename Matrix1::value_type;
  using value_type2 = typename Matrix2::value_type;

  size_type nrows = SMALL_MATRIX_NROWS, ncols = SMALL_MATRIX_NCOLS,
            nnnz = SMALL_MATRIX_NNZ;

  HostMatrix1 Ah(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(Ah, nrows, ncols, nnnz);
  Morpheus::Test::build_small_container(Ah);

  // Matrix1 A(nrows, ncols, nnnz);
  Matrix1 A;
  A.resize(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(A, nrows, ncols, nnnz);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_CSR_EMPTY(Bh);

  Bh.allocate(Ah);
  CHECK_CSR_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.row_offsets(2)    = 6;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = (value_type1)-3.33;

  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), 0);
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), 0);
    EXPECT_EQ(Bh.values(n), (value_type2)0);
  }

  // Now check device vector
  Matrix2 B;
  CHECK_CSR_EMPTY(B);

  Bh.row_offsets(1)    = 1;
  Bh.column_indices(2) = 2;
  Bh.values(1)         = (value_type2)-1.11;

  B.allocate(A);
  CHECK_CSR_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (size_type n = 0; n < nrows + 1; n++) {
    EXPECT_EQ(Bh.row_offsets(n), 0);
  }
  for (size_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.column_indices(n), 0);
    EXPECT_EQ(Bh.values(n), (value_type2)0);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_CSRMATRIX_BINARY_HPP