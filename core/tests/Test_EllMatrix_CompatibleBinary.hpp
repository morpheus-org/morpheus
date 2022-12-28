/**
 * Test_EllMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_ELLMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_ELLMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_EllMatrix.hpp>

using EllMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::EllMatrix<double>, types::compatible_types_set>::type;

using CompatibleEllMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        EllMatrixCompatibleTypes, EllMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleEllMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // EllMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // EllMatrix
  using host2   = typename type2::type::HostMirror;

  using SizeType = typename device1::size_type;

  CompatibleEllMatrixBinaryTest()
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
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Compatible Binary EllMatrix pairs
 *
 */
TYPED_TEST_SUITE(CompatibleEllMatrixBinaryTest, CompatibleEllMatrixBinary);

TYPED_TEST(CompatibleEllMatrixBinaryTest, ConstructionFromEllMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
            this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

TYPED_TEST(CompatibleEllMatrixBinaryTest, CopyAssignmentFromEllMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
            this->Aref.column_indices(), this->Aref.values());
  CHECK_ELL_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 this->nalign);
  CHECK_ELL_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                  this->nentries_per_row, this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_ELL_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh = Ah;
  CHECK_ELL_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.column_indices(1, 0) = 3;
  Ah.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_ELL_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_ELL_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->nnnz, this->nentries_per_row,
                 this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_ELL_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing construction of EllMatrix from \p DenseMatrix arrays.
 *
 */
TYPED_TEST(CompatibleEllMatrixBinaryTest, ConstructionFromDenseMatrix) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
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

  HostMatrix Ah_test(this->nrows, this->ncols, this->nnnz,
                     this->nentries_per_row, this->nalign);
  Morpheus::copy(A, Ah_test);

  VALIDATE_ELL_CONTAINER(Ah, Ah_test);

  Ah.column_indices(0, 1) = 1;
  Ah.values(0, 1)         = (value_type)-1.11;
  EXPECT_NE(Ah.column_indices(0, 1), Ah_test.column_indices(0, 1));
  EXPECT_NE(Ah.values(0, 1), Ah_test.values(0, 1));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_ELLMATRIX_COMPATIBLEBINARY_HPP