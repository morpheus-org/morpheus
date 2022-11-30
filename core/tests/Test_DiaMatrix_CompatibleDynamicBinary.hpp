/**
 * Test_DiaMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_DIAMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_DIAMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DiaMatrix.hpp>

using DiaMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DiaMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using DiaMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DiaMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and DiaMatrix
// containers
template <typename BinaryContainer>
class CompatibleDiaMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // DiaMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  using IndexType = typename device::index_type;

  CompatibleDiaMatrixDynamicTest()
      : nrows(3),
        ncols(3),
        nnnz(4),
        ndiag(4),
        nalign(32),
        Aref(3, 3, 4, 4),
        Ahref(3, 3, 4, 4) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  IndexType nrows, ncols, nnnz, ndiag, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary DiaMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleDiaMatrixDynamicTest,
                 DiaMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of DiaMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the DiaMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleDiaMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveDia) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using index_type        = typename Matrix::index_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference DiaMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is DiaMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_DIA_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.diagonal_offsets(2) = 2;
  Bh.values(1, 2)        = (value_type)-3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), Ch.diagonal_offsets(n));
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), Ch.values(i, j));
    }
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_DIA_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->nnnz, this->ndiag);
  Morpheus::copy(C, Ct);
  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), Ct.diagonal_offsets(n));
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), Ct.values(i, j));
    }
  }
}

/**
 * @brief Testing construction of DiaMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the DiaMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleDiaMatrixDynamicTest,
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
 * @brief Testing copy assignment of DiaMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the DiaMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleDiaMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveDia) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using index_type        = typename Matrix::index_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference DiaMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is DiaMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignment from dynamic
  HostMatrix Bh = Ah;
  CHECK_DIA_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.diagonal_offsets(2) = 2;
  Bh.values(1, 2)        = (value_type)-3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), Ch.diagonal_offsets(n));
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), Ch.values(i, j));
    }
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_DIA_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->nnnz, this->ndiag);
  Morpheus::copy(C, Ct);
  for (index_type n = 0; n < Bh.ndiags(); n++) {
    EXPECT_EQ(Bh.diagonal_offsets(n), Ct.diagonal_offsets(n));
  }
  for (index_type i = 0; i < Bh.values().nrows(); i++) {
    for (index_type j = 0; j < Bh.values().ncols(); j++) {
      EXPECT_EQ(Bh.values(i, j), Ct.values(i, j));
    }
  }
}

/**
 * @brief Testing copy assignment of DiaMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the DiaMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleDiaMatrixDynamicTest,
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
#endif  // TEST_CORE_TEST_DIAMATRIX_COMPATIBLEDYNAMICBINARY_HPP