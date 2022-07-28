/**
 * Test_CooMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_BINARY_HPP
#define TEST_CORE_TEST_COOMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>
#include <Macros.hpp>

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::types_set>::type;
using CooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixTypes, CooMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class CooMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CooMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CooMatrix
  using host2   = typename type2::type::HostMirror;

  CooMatrixBinaryTest() : Aref(3, 3, 4), Ahref(3, 3, 4) {}

  void SetUp() override {
    build_coomatrix(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary CooMatrix pairs
 *
 */
TYPED_TEST_CASE(CooMatrixBinaryTest, CooMatrixBinary);

TYPED_TEST(CooMatrixBinaryTest, ResizeFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using index_type  = typename Matrix1::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_type large_nrows = 500, large_ncols = 400, large_nnnz = 640;
  index_type small_nrows = 2, small_ncols = 3, small_nnnz = 2;

  Matrix1 A(nrows, ncols, nnnz);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  Matrix2 Alarge(large_nrows, large_ncols, large_nnnz);
  A.resize(Alarge);
  CHECK_COO_CONTAINERS(A, Alarge);

  HostMatrix1 Ah(large_nrows, large_ncols, large_nnnz);
  CHECK_COO_SIZES(Ah, large_nrows, large_ncols, large_nnnz);

  Morpheus::copy(A, Ah);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), this->Ahref.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), this->Ahref.column_indices(n));
    EXPECT_EQ(Ah.values(n), this->Ahref.values(n));
  }

  for (index_type n = nnnz; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.row_indices(n), 0);
    EXPECT_EQ(Ah.column_indices(n), 0);
    EXPECT_EQ(Ah.values(n), 0);
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.row_indices(1)    = 1;
  Ah.column_indices(2) = 10;
  Ah.values(0)         = -1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix1 Ahref_test(nrows, ncols, nnnz);
  Morpheus::copy(this->Ahref, Ahref_test);
  EXPECT_NE(Ah.row_indices(1), Ahref_test.row_indices(1));
  EXPECT_NE(Ah.column_indices(2), Ahref_test.column_indices(2));
  EXPECT_NE(Ah.values(0), Ahref_test.values(0));

  for (index_type n = nnnz; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.row_indices(n), 0);
    EXPECT_EQ(Ah.column_indices(n), 0);
    EXPECT_EQ(Ah.values(n), 0);
  }
  Matrix2 Asmall(small_nrows, small_ncols, small_nnnz);
  // Resize to smaller shape and non-zeros
  A.resize(Asmall);
  CHECK_COO_CONTAINERS(A, Asmall);
  Ah.resize(Asmall);
  CHECK_COO_CONTAINERS(Ah, Asmall);

  // Set back to normal
  Ah.row_indices(1)    = 0;
  Ah.column_indices(2) = 1;
  Ah.values(0)         = 1.11;

  Morpheus::copy(Ah, A);
  for (index_type n = 0; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.row_indices(n), Ahref_test.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ahref_test.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref_test.values(n));
  }
}

/**
 * @brief Testing allocation of CooMatrix container from another CooMatrix
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(CooMatrixBinaryTest, AllocateFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;

  HostMatrix1 Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);
  build_coomatrix(Ah);

  Matrix1 A(nrows, ncols, nnnz);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_COO_SIZES(Bh, 0, 0, 0);

  Bh.allocate(Ah);
  CHECK_COO_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), 0);
    EXPECT_EQ(Bh.column_indices(n), 0);
    EXPECT_EQ(Bh.values(n), 0);
  }

  // Now check device vector
  Matrix2 B;
  CHECK_COO_SIZES(B, 0, 0, 0);

  Bh.row_indices(1)    = 1;
  Bh.column_indices(2) = 2;
  Bh.values(1)         = -1.11;

  B.allocate(A);
  CHECK_COO_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), 0);
    EXPECT_EQ(Bh.column_indices(n), 0);
    EXPECT_EQ(Bh.values(n), 0);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_COOMATRIX_BINARY_HPP