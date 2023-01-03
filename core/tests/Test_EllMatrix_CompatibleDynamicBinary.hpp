/**
 * Test_EllMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_ELLMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_ELLMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_EllMatrix.hpp>

using EllMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::EllMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using EllMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        EllMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and EllMatrix
// containers
template <typename BinaryContainer>
class CompatibleEllMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // EllMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  using SizeType = typename device::size_type;

  CompatibleEllMatrixDynamicTest()
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
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary EllMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleEllMatrixDynamicTest,
                 EllMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of EllMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the EllMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleEllMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveEll) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference EllMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is EllMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_ELL_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.column_indices(1, 0) = 3;
  Bh.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), Ch.column_indices(i, j));
      EXPECT_EQ(Bh.values(i, j), Ch.values(i, j));
    }
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_ELL_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), Ct.column_indices(i, j));
      EXPECT_EQ(Bh.values(i, j), Ct.values(i, j));
    }
  }
}

/**
 * @brief Testing construction of EllMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the EllMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleEllMatrixDynamicTest,
           ConstructionFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B(A), Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh(Ah), Morpheus::RuntimeException);
}

/**
 * @brief Testing copy assignment of EllMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the EllMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleEllMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveEll) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference EllMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is EllMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignment from dynamic
  HostMatrix Bh = Ah;
  CHECK_ELL_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.column_indices(1, 0) = 3;
  Bh.values(0, 1)         = (value_type)-3.33;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), Ch.column_indices(i, j));
      EXPECT_EQ(Bh.values(i, j), Ch.values(i, j));
    }
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_ELL_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->nnnz, this->nentries_per_row);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.values().nrows(); i++) {
    for (size_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.column_indices(i, j), Ct.column_indices(i, j));
      EXPECT_EQ(Bh.values(i, j), Ct.values(i, j));
    }
  }
}

/**
 * @brief Testing copy assignment of EllMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the EllMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleEllMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixDifferentActive) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;

  DynamicMatrix A;
  A.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(A.active_index(), this->Aref.format_index());
  EXPECT_THROW(Matrix B = A, Morpheus::RuntimeException);

  DynamicHostMatrix Ah;
  Ah.activate(Morpheus::COO_FORMAT);
  EXPECT_NE(Ah.active_index(), this->Ahref.format_index());
  EXPECT_THROW(HostMatrix Bh = Ah, Morpheus::RuntimeException);
}

}  // namespace Test
#endif  // TEST_CORE_TEST_ELLMATRIX_COMPATIBLEDYNAMICBINARY_HPP