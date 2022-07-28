/**
 * Test_CooMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>

using CooMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CooMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using CooMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

/**
 * @brief Checks the sizes of a CooMatrix container against a number of rows,
 * columns and non-zeros
 *
 */
#define CHECK_COO_SIZES(A, num_rows, num_cols, num_nnz) \
  {                                                     \
    EXPECT_EQ(A.nrows(), num_rows);                     \
    EXPECT_EQ(A.ncols(), num_cols);                     \
    EXPECT_EQ(A.nnnz(), num_nnz);                       \
    EXPECT_EQ(A.row_indices().size(), num_nnz);         \
    EXPECT_EQ(A.column_indices().size(), num_nnz);      \
    EXPECT_EQ(A.values().size(), num_nnz);              \
  }

/**
 * @brief Checks the sizes of two CooMatrix containers if they match
 *
 */
#define CHECK_COO_CONTAINERS(A, B)                                   \
  {                                                                  \
    EXPECT_EQ(A.nrows(), B.nrows());                                 \
    EXPECT_EQ(A.ncols(), B.ncols());                                 \
    EXPECT_EQ(A.nnnz(), B.nnnz());                                   \
    EXPECT_EQ(A.row_indices().size(), B.row_indices().size());       \
    EXPECT_EQ(A.column_indices().size(), B.column_indices().size()); \
    EXPECT_EQ(A.values().size(), B.values().size());                 \
  }

/**
 * @brief Checks if the data arrays of two CooMatrix containers contain the same
 * data.
 *
 */
#define VALIDATE_COO_CONTAINER(A, Aref, nnnz, type)           \
  {                                                           \
    for (type n = 0; n < nnnz; n++) {                         \
      EXPECT_EQ(A.row_indices(n), Aref.row_indices(n));       \
      EXPECT_EQ(A.column_indices(n), Aref.column_indices(n)); \
      EXPECT_EQ(A.values(n), Aref.values(n));                 \
    }                                                         \
  }

/**
 * @brief Builds a sample CooMatrix container. Assumes we have already
 * constructed the matrix and we are only adding data.
 *
 * @tparam Matrix A CooMatrix type
 * @param A The CooMatrix we will be initializing.
 */
template <typename Matrix>
void build_coomatrix(Matrix& A) {
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  CHECK_COO_SIZES(A, 3, 3, 4);

  // clang-format off
  A.row_indices(0) = 0; A.column_indices(0) = 0; A.values(0) = 1.11;
  A.row_indices(1) = 0; A.column_indices(1) = 2; A.values(1) = 2.22;
  A.row_indices(2) = 1; A.column_indices(2) = 2; A.values(2) = 3.33;
  A.row_indices(3) = 2; A.column_indices(3) = 1; A.values(3) = 4.44;
  // clang-format on
}

// Used for testing compatible binary operations between Dynamic and CooMatrix
// containers
template <typename BinaryContainer>
class CompatibleCooMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // CooMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  CompatibleCooMatrixDynamicTest() : Aref(3, 3, 4), Ahref(3, 3, 4) {}

  void SetUp() override {
    build_coomatrix(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary CooMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_CASE(CompatibleCooMatrixDynamicTest,
                CooMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of CooMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the CooMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveCoo) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using index_type        = typename Matrix::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the reference CooMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CooMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_COO_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.row_indices(2)    = 2;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ch.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_COO_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ct.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing construction of CooMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the CooMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           ConstructionFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B(A), Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh(Ah), Morpheus::RuntimeException);
}

/**
 * @brief Testing copy assignment of CooMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the CooMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveCoo) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using index_type        = typename Matrix::index_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  // Build matrix from the reference CooMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is CooMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignemnt from dynamic
  HostMatrix Bh = Ah;
  CHECK_COO_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.row_indices(2)    = 2;
  Bh.column_indices(1) = 1;
  Bh.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ch.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ch.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ch.values(n));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_COO_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(nrows, ncols, nnnz);
  Morpheus::copy(C, Ct);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ct.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ct.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ct.values(n));
  }
}

/**
 * @brief Testing copy assignment of CooMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the CooMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleCooMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B = A, Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::CSR_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh = Ah, Morpheus::RuntimeException);
}

}  // namespace Test
#endif  // TEST_CORE_TEST_COOMATRIX_COMPATIBLEDYNAMICBINARY_HPP