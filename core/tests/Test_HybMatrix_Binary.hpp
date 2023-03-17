/**
 * Test_HybMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_HYBMATRIX_BINARY_HPP
#define TEST_CORE_TEST_HYBMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HybMatrix.hpp>

using HybMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HybMatrix<double>,
                                               types::types_set>::type;
using HybMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HybMatrixTypes, HybMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class HybMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // HybMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // HybMatrix
  using host2   = typename type2::type::HostMirror;

  using IndexType = typename device1::size_type;

  HybMatrixBinaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        ell_nnnz(SMALL_HYB_ELL_NNZ),
        coo_nnnz(SMALL_HYB_COO_NNZ),
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

  IndexType nrows, ncols, ell_nnnz, coo_nnnz, nentries_per_row, nalign;
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary HybMatrix pairs
 *
 */
TYPED_TEST_SUITE(HybMatrixBinaryTest, HybMatrixBinary);

TYPED_TEST(HybMatrixBinaryTest, ResizeFromHybMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using size_type   = typename Matrix1::size_type;
  using value_type  = typename Matrix1::value_type;

  Matrix1 A(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
            this->nentries_per_row);
  CHECK_HYB_SIZES(A, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(this->Ahref, A);

  // Resize to larger shape and non-zeros
  size_type large_nrows = 500, large_ncols = 500, large_ell_nnnz = 640,
            large_coo_nnnz = 340, large_nentries_per_row = 110;
  Matrix2 Alarge(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                 large_nentries_per_row);
  A.resize(Alarge);
  CHECK_HYB_CONTAINERS(Alarge, A);

  HostMatrix1 Ah(large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                 large_nentries_per_row);
  CHECK_HYB_SIZES(Ah, large_nrows, large_ncols, large_ell_nnnz, large_coo_nnnz,
                  large_nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);

  for (size_type i = 0; i < this->Ahref.ell().values().nrows(); i++) {
    for (size_type j = 0; j < this->Ahref.ell().values().ncols(); j++) {
      EXPECT_EQ(Ah.ell().column_indices(i, j),
                this->Ahref.ell().column_indices(i, j));
      EXPECT_EQ(Ah.ell().values(i, j), this->Ahref.ell().values(i, j));
    }
  }

  for (size_type i = this->Ahref.ell().values().nrows();
       i < Ah.ell().values().nrows(); i++) {
    for (size_type j = this->Ahref.ell().values().ncols();
         j < Ah.ell().values().ncols(); j++) {
      EXPECT_EQ(Ah.ell().column_indices(i, j), 0);
      EXPECT_EQ(Ah.ell().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < this->Ahref.coo().values().size(); i++) {
    EXPECT_EQ(Ah.coo().row_indices(i), this->Ahref.coo().row_indices(i));
    EXPECT_EQ(Ah.coo().column_indices(i), this->Ahref.coo().column_indices(i));
    EXPECT_EQ(Ah.coo().values(i), this->Ahref.coo().values(i));
  }

  for (size_type i = this->Ahref.coo().values().size();
       i < Ah.coo().values().size(); i++) {
    EXPECT_EQ(Ah.coo().row_indices(i), 0);
    EXPECT_EQ(Ah.coo().column_indices(i), 0);
    EXPECT_EQ(Ah.coo().values(i), 0);
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.ell().column_indices(1, 0) = 2;
  Ah.ell().values(1, 0)         = (value_type)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type)-1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix1 Ahref_test(this->nrows, this->ncols, this->ell_nnnz,
                         this->coo_nnnz, this->nentries_per_row);
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
  size_type small_nrows = 2, small_ncols = 2, small_ell_nnnz = 2,
            small_coo_nnnz = 1, small_nentries_per_row = 2;
  Matrix2 Asmall(small_nrows, small_ncols, small_ell_nnnz, small_coo_nnnz,
                 small_nentries_per_row);
  A.resize(Asmall);
  CHECK_HYB_CONTAINERS(Asmall, A);
  Ah.resize(Asmall);
  CHECK_HYB_CONTAINERS(Asmall, Ah);

  // Set back to normal
  Ah.ell().column_indices(1, 0) = 1;
  Ah.ell().values(1, 0)         = (value_type)5.55;
  Ah.coo().row_indices(0)       = 0;
  Ah.coo().column_indices(0)    = 8;
  Ah.coo().values(0)            = (value_type)4.44;
  Morpheus::copy(Ah, A);

  VALIDATE_HYB_CONTAINER(Ah, Ahref_test);
}

/**
 * @brief Testing allocation of HybMatrix container from another HybMatrix
 * container with the different parameters. New allocation shouldn't alias
 the
 * original.
 *
 */
TYPED_TEST(HybMatrixBinaryTest, AllocateFromHybMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using size_type   = typename Matrix1::size_type;
  using value_type1 = typename Matrix1::value_type;
  using value_type2 = typename Matrix2::value_type;

  HostMatrix1 Ah(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                 this->nentries_per_row);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::Test::build_small_container(Ah);

  Matrix1 A(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
            this->nentries_per_row);
  CHECK_HYB_SIZES(A, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);
  Morpheus::copy(Ah, A);

  HostMatrix2 Bh;
  CHECK_HYB_EMPTY(Bh);

  Bh.allocate(Ah);
  CHECK_HYB_CONTAINERS(Ah, Bh);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 2;
  Ah.ell().values(1, 0)         = (value_type1)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type1)-1.11;

  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), 0);
      EXPECT_EQ(Bh.ell().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), 0);
    EXPECT_EQ(Bh.coo().column_indices(i), 0);
    EXPECT_EQ(Bh.coo().values(i), 0);
  }

  // Now check device vector
  Matrix2 B;
  CHECK_HYB_EMPTY(B);

  Bh.ell().column_indices(1, 0) = 2;
  Bh.ell().values(1, 0)         = (value_type2)-1.11;
  Bh.coo().row_indices(0)       = 1;
  Bh.coo().column_indices(0)    = 1;
  Bh.coo().values(0)            = (value_type2)-1.11;

  B.allocate(A);
  CHECK_HYB_CONTAINERS(A, B);
  Morpheus::copy(B, Bh);

  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), 0);
      EXPECT_EQ(Bh.ell().values(i, j), 0);
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), 0);
    EXPECT_EQ(Bh.coo().column_indices(i), 0);
    EXPECT_EQ(Bh.coo().values(i), 0);
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_HYBMATRIX_BINARY_HPP