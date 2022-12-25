/**
 * Test_HybMatrix_CompatibleDynamicBinary.hpp
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

#ifndef TEST_CORE_TEST_HYBMATRIX_COMPATIBLEDYNAMICBINARY_HPP
#define TEST_CORE_TEST_HYBMATRIX_COMPATIBLEDYNAMICBINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_HybMatrix.hpp>

using HybMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::HybMatrix<double>, types::compatible_types_set>::type;

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using HybMatrixDynamicCompatibleBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        HybMatrixCompatibleTypes, DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations between Dynamic and HybMatrix
// containers
template <typename BinaryContainer>
class CompatibleHybMatrixDynamicTest : public ::testing::Test {
 public:
  using type     = BinaryContainer;
  using concrete = typename BinaryContainer::type1;  // Unary
  using dynamic  = typename BinaryContainer::type2;  // Unary

  using device = typename concrete::type;  // HybMatrix
  using host   = typename concrete::type::HostMirror;

  using dynamic_device = typename dynamic::type;  // DynamicMatrix
  using dynamic_host   = typename dynamic::type::HostMirror;

  using SizeType = typename device::size_type;

  CompatibleHybMatrixDynamicTest()
      : nrows(3),
        ncols(3),
        ell_nnnz(3),
        coo_nnnz(1),
        nentries_per_row(1),
        nalign(32),
        Aref(3, 3, 3, 1, 1),
        Ahref(3, 3, 3, 1, 1) {}

  void SetUp() override {
    Morpheus::Test::build_small_container(Ahref);

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  SizeType nrows, ncols, ell_nnnz, coo_nnnz, nentries_per_row, nalign;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Binary HybMatrix-DynamicMatrix Compatible
 pairs
 *
 */
TYPED_TEST_SUITE(CompatibleHybMatrixDynamicTest,
                 HybMatrixDynamicCompatibleBinary);

/**
 * @brief Testing construction of HybMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * the same as the HybMatrix format so construction should succeed.
 *
 */
TYPED_TEST(CompatibleHybMatrixDynamicTest,
           ConstructionFromDynamicMatrixActiveHyb) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference HybMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is HybMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy construction from dynamic
  HostMatrix Bh(Ah);
  CHECK_HYB_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch(Ah);

  // Change values in one container
  Bh.ell().column_indices(1, 0) = 1;
  Bh.ell().values(1, 0)         = (value_type)-1.11;
  Bh.coo().row_indices(0)       = 1;
  Bh.coo().column_indices(0)    = 1;
  Bh.coo().values(0)            = (value_type)-1.11;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), Ch.ell().column_indices(i, j));
      EXPECT_EQ(Bh.ell().values(i, j), Ch.ell().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), Ch.coo().row_indices(i));
    EXPECT_EQ(Bh.coo().column_indices(i), Ch.coo().column_indices(i));
    EXPECT_EQ(Bh.coo().values(i), Ch.coo().values(i));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_HYB_CONTAINERS(B, this->Aref);

  Matrix C(A);
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                this->nentries_per_row);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), Ct.ell().column_indices(i, j));
      EXPECT_EQ(Bh.ell().values(i, j), Ct.ell().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), Ct.coo().row_indices(i));
    EXPECT_EQ(Bh.coo().column_indices(i), Ct.coo().column_indices(i));
    EXPECT_EQ(Bh.coo().values(i), Ct.coo().values(i));
  }
}

/**
 * @brief Testing construction of HybMatrix container from another DynamicMatrix
 * container with different parameters. Active type of the DynamicMatrix is
 * different from the HybMatrix format so construction should fail.
 *
 */
TYPED_TEST(CompatibleHybMatrixDynamicTest,
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
 * @brief Testing copy assignment of HybMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is the same as the HybMatrix format so assignment should
 * succeed.
 *
 */
TYPED_TEST(CompatibleHybMatrixDynamicTest,
           CopyAssignmentFromDynamicMatrixActiveHyb) {
  using Matrix            = typename TestFixture::device;
  using HostMatrix        = typename TestFixture::host;
  using DynamicMatrix     = typename TestFixture::dynamic_device;
  using DynamicHostMatrix = typename TestFixture::dynamic_host;
  using size_type         = typename Matrix::size_type;
  using value_type        = typename Matrix::value_type;

  // Build matrix from the reference HybMatrix
  DynamicMatrix A(this->Aref);
  DynamicHostMatrix Ah(this->Ahref);

  // Active type is HybMatrix
  EXPECT_EQ(A.active_index(), this->Aref.format_index());

  // Copy assignment from dynamic
  HostMatrix Bh = Ah;
  CHECK_HYB_CONTAINERS(Bh, this->Ahref);

  // Copy to check alias
  HostMatrix Ch = Ah;

  // Change values in one container
  Bh.ell().column_indices(1, 0) = 1;
  Bh.ell().values(1, 0)         = (value_type)-1.11;
  Bh.coo().row_indices(0)       = 1;
  Bh.coo().column_indices(0)    = 1;
  Bh.coo().values(0)            = (value_type)-1.11;

  // Other container should reflect the same changes
  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), Ch.ell().column_indices(i, j));
      EXPECT_EQ(Bh.ell().values(i, j), Ch.ell().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), Ch.coo().row_indices(i));
    EXPECT_EQ(Bh.coo().column_indices(i), Ch.coo().column_indices(i));
    EXPECT_EQ(Bh.coo().values(i), Ch.coo().values(i));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_HYB_CONTAINERS(B, this->Aref);

  Matrix C = A;
  Morpheus::copy(Bh, B);

  // Send other vector back to host for check
  HostMatrix Ct(this->nrows, this->ncols, this->ell_nnnz, this->coo_nnnz,
                this->nentries_per_row);
  Morpheus::copy(C, Ct);
  for (size_type i = 0; i < Bh.ell().values().nrows(); i++) {
    for (size_type j = 0; j < Bh.ell().values().ncols(); j++) {
      EXPECT_EQ(Bh.ell().column_indices(i, j), Ct.ell().column_indices(i, j));
      EXPECT_EQ(Bh.ell().values(i, j), Ct.ell().values(i, j));
    }
  }

  for (size_type i = 0; i < Bh.coo().values().size(); i++) {
    EXPECT_EQ(Bh.coo().row_indices(i), Ct.coo().row_indices(i));
    EXPECT_EQ(Bh.coo().column_indices(i), Ct.coo().column_indices(i));
    EXPECT_EQ(Bh.coo().values(i), Ct.coo().values(i));
  }
}

/**
 * @brief Testing copy assignment of HybMatrix container from another
 * DynamicMatrix container with different parameters. Active type of the
 * DynamicMatrix is different from the HybMatrix format so assignment should
 * fail.
 *
 */
TYPED_TEST(CompatibleHybMatrixDynamicTest,
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
#endif  // TEST_CORE_TEST_HYBMATRIX_COMPATIBLEDYNAMICBINARY_HPP