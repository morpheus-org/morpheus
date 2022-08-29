/**
 * Test_DiaMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_DIAMATRIX_BINARY_HPP
#define TEST_CORE_TEST_DIAMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DiaMatrix.hpp>

using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::types_set>::type;
using DiaMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DiaMatrixTypes, DiaMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class DiaMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // DiaMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // DiaMatrix
  using host2   = typename type2::type::HostMirror;

  using IndexType = typename device1::index_type;

  DiaMatrixBinaryTest()
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
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary DiaMatrix pairs
 *
 */
TYPED_TEST_SUITE(DiaMatrixBinaryTest, DiaMatrixBinary);

TYPED_TEST(DiaMatrixBinaryTest, ResizeFromDiaMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using index_type  = typename Matrix1::index_type;
  using value_type  = typename Matrix1::value_type;

  Matrix1 A(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(A, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  index_type large_nrows = 500, large_ncols = 500, large_nnnz = 640,
             large_ndiag = 110;
  Matrix2 Alarge(large_nrows, large_ncols, large_nnnz, large_ndiag);
  A.resize(Alarge);
  CHECK_DIA_CONTAINERS(Alarge, A);

  HostMatrix1 Ah(large_nrows, large_ncols, large_nnnz, large_ndiag);
  CHECK_DIA_SIZES(Ah, large_nrows, large_ncols, large_nnnz, large_ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  for (index_type n = 0; n < this->ndiag; n++) {
    EXPECT_EQ(Ah.diagonal_offsets(n), this->Ahref.diagonal_offsets(n));
  }
  for (index_type i = 0; i < this->Ahref.values().nrows(); i++) {
    for (index_type j = 0; j < this->Ahref.values().ncols(); j++) {
      EXPECT_EQ(Ah.values(i, j), this->Ahref.values(i, j));
    }
  }

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
  Ah.values(0, 1)        = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix1 Ahref_test(this->nrows, this->ncols, this->nnnz, this->ndiag);
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
  index_type small_nrows = 2, small_ncols = 2, small_nnnz = 2, small_ndiag = 2;
  Matrix2 Asmall(small_nrows, small_ncols, small_nnnz, small_ndiag);
  A.resize(Asmall);
  CHECK_DIA_CONTAINERS(Asmall, A);
  Ah.resize(Asmall);
  CHECK_DIA_CONTAINERS(Asmall, Ah);

  // Set back to normal
  Ah.diagonal_offsets(1) = 0;
  Ah.values(0, 1)        = (value_type)1.11;
  Morpheus::copy(Ah, A);

  VALIDATE_DIA_CONTAINER(Ah, Ahref_test, index_type);
}

/**
 * @brief Testing allocation of DiaMatrix container from another DiaMatrix
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(DiaMatrixBinaryTest, AllocateFromDiaMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;
  using value_type1 = typename Matrix1::value_type;
  using value_type2 = typename Matrix2::value_type;

  HostMatrix1 Ah(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  build_diamatrix(Ah);

  Matrix1 A(this->nrows, this->ncols, this->nnnz, this->ndiag);
  CHECK_DIA_SIZES(A, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_DIA_EMPTY(Bh);

  Bh.allocate(Ah);
  CHECK_DIA_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.diagonal_offsets(1) = 1;
  Ah.values(0, 1)        = (value_type1)-1.11;

  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), 0);
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), 0);
    }
  }

  // Now check device vector
  Matrix2 B;
  CHECK_DIA_EMPTY(B);

  Bh.diagonal_offsets(1) = 1;
  Bh.values(0, 1)        = (value_type2)-1.11;

  B.allocate(A);
  CHECK_DIA_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), 0);
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), 0);
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DIAMATRIX_BINARY_HPP