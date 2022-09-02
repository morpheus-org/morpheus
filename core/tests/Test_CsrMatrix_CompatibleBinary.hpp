/**
 * Test_CsrMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_CSRMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_CSRMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_CsrMatrix.hpp>

using CsrMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CsrMatrix<double>, types::compatible_types_set>::type;

using CompatibleCsrMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CsrMatrixCompatibleTypes, CsrMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleCsrMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CsrMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CsrMatrix
  using host2   = typename type2::type::HostMirror;

  CompatibleCsrMatrixBinaryTest() : Aref(3, 3, 4), Ahref(3, 3, 4) {}

  void SetUp() override {
    build_csrmatrix(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  device1 Aref;
  host1 Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Compatible Binary CsrMatrix pairs
 *
 */
TYPED_TEST_SUITE(CompatibleCsrMatrixBinaryTest, CompatibleCsrMatrixBinary);

TYPED_TEST(CompatibleCsrMatrixBinaryTest, ConstructionFromCsrMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;
  using value_type2 = typename Matrix2::value_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix2 A(nrows, ncols, nnnz, this->Aref.row_offsets(),
            this->Aref.column_indices(), this->Aref.values());
  CHECK_CSR_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_CSR_CONTAINER(Ah, this->Ahref, nrows, nnnz, index_type);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_CSR_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_offsets(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = (value_type2)-3.33;

  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Bh, Ah, nrows, nnnz, index_type);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_CSR_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  VALIDATE_CSR_CONTAINER(Bt, Ah, nrows, nnnz, index_type);
}

TYPED_TEST(CompatibleCsrMatrixBinaryTest, CopyAssignmentFromCsrMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;
  using value_type2 = typename Matrix2::value_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix2 A(nrows, ncols, nnnz, this->Aref.row_offsets(),
            this->Aref.column_indices(), this->Aref.values());
  CHECK_CSR_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_CSR_CONTAINER(Ah, this->Ahref, nrows, nnnz, index_type);

  // Default copy asssignment
  HostMatrix1 Bh = Ah;
  CHECK_CSR_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_offsets(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = (value_type2)-3.33;

  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Bh, Ah, nrows, nnnz, index_type);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_CSR_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  VALIDATE_CSR_CONTAINER(Bt, Ah, nrows, nnnz, index_type);
}

/**
 * @brief Testing construction of CsrMatrix from \p DenseVector arrays.
 *
 */
TYPED_TEST(CompatibleCsrMatrixBinaryTest, ConstructionFromDenseVector) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, this->Aref.row_offsets(),
           this->Aref.column_indices(), this->Aref.values());
  CHECK_CSR_CONTAINERS(A, this->Aref);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_CSR_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_CSR_CONTAINER(Ah, this->Ahref, nrows, nnnz, index_type);

  HostMatrix Ah_test(nrows, ncols, nnnz);
  Morpheus::copy(A, Ah_test);

  VALIDATE_CSR_CONTAINER(Ah, Ah_test, nrows, nnnz, index_type);

  Ah.row_offsets(2) = 2;
  EXPECT_NE(Ah.row_offsets(2), Ah_test.row_offsets(2));
  Ah.column_indices(1) = 1;
  EXPECT_NE(Ah.column_indices(1), Ah_test.column_indices(1));
  Ah.values(0) = (value_type)-1.11;
  EXPECT_NE(Ah.values(0), Ah_test.values(0));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_CSRMATRIX_COMPATIBLEBINARY_HPP