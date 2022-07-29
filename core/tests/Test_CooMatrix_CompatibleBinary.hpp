/**
 * Test_CooMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_COOMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>
#include <Macros.hpp>

using CooMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CooMatrix<double>, types::compatible_types_set>::type;

using CompatibleCooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixCompatibleTypes, CooMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleCooMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CooMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CooMatrix
  using host2   = typename type2::type::HostMirror;

  CompatibleCooMatrixBinaryTest() : Aref(3, 3, 4), Ahref(3, 3, 4) {}

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
 * @brief Test Suite using the Compatible Binary CooMatrix pairs
 *
 */
TYPED_TEST_CASE(CompatibleCooMatrixBinaryTest, CompatibleCooMatrixBinary);

TYPED_TEST(CompatibleCooMatrixBinaryTest, ConstructionFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix2 A(nrows, ncols, nnnz, this->Aref.row_indices(),
            this->Aref.column_indices(), this->Aref.values());
  CHECK_COO_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_COO_CONTAINER(Ah, this->Ahref, nnnz, index_type);

  // Default copy construction
  HostMatrix1 Bh(Ah);
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Bh, Ah, nnnz, index_type);

  // Now check device Matrix
  Matrix1 B(A);
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  VALIDATE_COO_CONTAINER(Bt, Ah, nnnz, index_type);
}

TYPED_TEST(CompatibleCooMatrixBinaryTest, CopyAssignmentFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix2 A(nrows, ncols, nnnz, this->Aref.row_indices(),
            this->Aref.column_indices(), this->Aref.values());
  CHECK_COO_CONTAINERS(A, this->Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_COO_CONTAINER(Ah, this->Ahref, nnnz, index_type);

  // Default copy asssignment
  HostMatrix1 Bh = Ah;
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Bh, Ah, nnnz, index_type);

  // Now check device Matrix
  Matrix1 B = A;
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  VALIDATE_COO_CONTAINER(Bt, Ah, nnnz, index_type);
}

/**
 * @brief Testing construction of CooMatrix from \p DenseVector arrays.
 *
 */
TYPED_TEST(CompatibleCooMatrixBinaryTest, ConstructionFromDenseVector) {
  using Matrix     = typename TestFixture::device2;
  using HostMatrix = typename TestFixture::host2;
  using index_type = typename Matrix::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, this->Aref.row_indices(),
           this->Aref.column_indices(), this->Aref.values());
  CHECK_COO_CONTAINERS(A, this->Aref);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);

  Morpheus::copy(A, Ah);
  VALIDATE_COO_CONTAINER(Ah, this->Ahref, nnnz, index_type);

  HostMatrix Ah_test(nrows, ncols, nnnz);
  Morpheus::copy(A, Ah_test);

  VALIDATE_COO_CONTAINER(Ah, Ah_test, nnnz, index_type);

  Ah.row_indices(2) = 2;
  EXPECT_NE(Ah.row_indices(2), Ah_test.row_indices(2));
  Ah.column_indices(1) = 1;
  EXPECT_NE(Ah.column_indices(1), Ah_test.column_indices(1));
  Ah.values(0) = -1.11;
  EXPECT_NE(Ah.values(0), Ah_test.values(0));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_COOMATRIX_COMPATIBLEBINARY_HPP