/**
 * Test_HybMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_HYBMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_HYBMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_HybMatrix.hpp>

using HybMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::HybMatrix<double>, types::compatible_types_set>::type;

using CompatibleHybMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HybMatrixCompatibleTypes, HybMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleHybMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // HybMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // HybMatrix
  using host2   = typename type2::type::HostMirror;

  using SizeType = typename device1::size_type;

  CompatibleHybMatrixBinaryTest()
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

  SizeType nrows, ncols, ell_nnnz, coo_nnnz, nentries_per_row, nalign;
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Compatible Binary HybMatrix pairs
 *
 */
TYPED_TEST_SUITE(CompatibleHybMatrixBinaryTest, CompatibleHybMatrixBinary);

TYPED_TEST(CompatibleHybMatrixBinaryTest, ConstructionFromHybMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                 this->nentries_per_row, this->nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 2;
  Ah.ell().values(1, 0)         = (value_type)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type)-1.11;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                 this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

TYPED_TEST(CompatibleHybMatrixBinaryTest, CopyAssignmentFromHybMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                 this->nentries_per_row, this->nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh = Ah;
  CHECK_HYB_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.ell().column_indices(1, 0) = 2;
  Ah.ell().values(1, 0)         = (value_type)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type)-1.11;

  // Other container should reflect the same changes
  VALIDATE_HYB_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_HYB_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                 this->nentries_per_row, this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_HYB_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing construction of HybMatrix from \p CooMatrix and \p EllMatrix.
 *
 */
TYPED_TEST(CompatibleHybMatrixBinaryTest, ConstructionFromDenseMatrix) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->Aref.ell(), this->Aref.coo());
  CHECK_HYB_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                this->nentries_per_row, this->nalign);
  CHECK_HYB_SIZES(Ah, this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_HYB_CONTAINER(Ah, this->Ahref);

  HostMatrix Ah_test(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                     this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah_test);

  VALIDATE_HYB_CONTAINER(Ah, Ah_test);

  Ah.ell().column_indices(1, 0) = 2;
  Ah.ell().values(1, 0)         = (value_type)-1.11;
  Ah.coo().row_indices(0)       = 1;
  Ah.coo().column_indices(0)    = 1;
  Ah.coo().values(0)            = (value_type)-1.11;
  EXPECT_NE(Ah.ell().column_indices(1, 0), Ah_test.ell().column_indices(1, 0));
  EXPECT_NE(Ah.ell().values(1, 0), Ah_test.ell().values(1, 0));
  EXPECT_NE(Ah.coo().row_indices(0), Ah_test.coo().row_indices(0));
  EXPECT_NE(Ah.coo().column_indices(0), Ah_test.coo().column_indices(0));
  EXPECT_NE(Ah.coo().values(0), Ah_test.coo().values(0));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_HYBMATRIX_COMPATIBLEBINARY_HPP