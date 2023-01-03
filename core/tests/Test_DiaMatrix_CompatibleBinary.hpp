/**
 * Test_DiaMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_DIAMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_DIAMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_DiaMatrix.hpp>

using DiaMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DiaMatrix<double>, types::compatible_types_set>::type;

using CompatibleDiaMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DiaMatrixCompatibleTypes, DiaMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleDiaMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // DiaMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // DiaMatrix
  using host2   = typename type2::type::HostMirror;

  using SizeType = typename device1::size_type;

  CompatibleDiaMatrixBinaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        nnnz(SMALL_MATRIX_NNZ),
        ndiag(SMALL_DIA_MATRIX_NDIAGS),
        nalign(SMALL_MATRIX_ALIGNMENT),
        Aref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
             SMALL_DIA_MATRIX_NDIAGS, SMALL_MATRIX_ALIGNMENT),
        Ahref(SMALL_MATRIX_NROWS, SMALL_MATRIX_NCOLS, SMALL_MATRIX_NNZ,
              SMALL_DIA_MATRIX_NDIAGS, SMALL_MATRIX_ALIGNMENT) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  SizeType nrows, ncols, nnnz, ndiag, nalign;
  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Compatible Binary DiaMatrix pairs
 *
 */
TYPED_TEST_SUITE(CompatibleDiaMatrixBinaryTest, CompatibleDiaMatrixBinary);

TYPED_TEST(CompatibleDiaMatrixBinaryTest, ConstructionFromDiaMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
            this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                 this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(1) = 7;
  Ah.values(8, 0)        = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                 this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah);
}

TYPED_TEST(CompatibleDiaMatrixBinaryTest, CopyAssignmentFromDiaMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix2::value_type;

  // Build matrix from the device vectors
  Matrix2 A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
            this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                 this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref);

  // Default copy construction
  HostMatrix1 Bh = Ah;
  CHECK_DIA_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.diagonal_offsets(1) = 7;
  Ah.values(8, 0)        = (value_type)-3.33;

  // Other container should reflect the same changes
  VALIDATE_DIA_CONTAINER(Bh, Ah);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_DIA_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(this->nrows, this->ncols, this->nnnz, this->ndiag,
                 this->nalign);
  Morpheus::copy(B, Bt);
  VALIDATE_DIA_CONTAINER(Bt, Ah);
}

/**
 * @brief Testing construction of DiaMatrix from \p DenseVector and
 * \p DenseMatrix arrays.
 *
 */
TYPED_TEST(CompatibleDiaMatrixBinaryTest, ConstructionFromDenseVector) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
  using value_type = typename Matrix::value_type;

  // Build matrix from the device vectors
  Matrix A(this->nrows, this->ncols, this->nnnz, this->Aref.diagonal_offsets(),
           this->Aref.values());
  CHECK_DIA_CONTAINERS(A, this->Aref);

  HostMatrix Ah(this->nrows, this->ncols, this->nnnz, this->ndiag,
                this->nalign);
  CHECK_DIA_SIZES(Ah, this->nrows, this->ncols, this->nnnz, this->ndiag,
                  this->nalign);

  Morpheus::copy(A, Ah);
  VALIDATE_DIA_CONTAINER(Ah, this->Ahref);

  HostMatrix Ah_test(this->nrows, this->ncols, this->nnnz, this->ndiag,
                     this->nalign);
  Morpheus::copy(A, Ah_test);

  VALIDATE_DIA_CONTAINER(Ah, Ah_test);

  Ah.diagonal_offsets(1) = 7;
  Ah.values(8, 0)        = (value_type)-3.33;
  EXPECT_NE(Ah.diagonal_offsets(1), Ah_test.diagonal_offsets(1));
  EXPECT_NE(Ah.values(8, 0), Ah_test.values(8, 0));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DIAMATRIX_COMPATIBLEBINARY_HPP