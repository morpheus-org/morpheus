/**
 * Test_DynamicMatrix.hpp
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

#ifndef TEST_CORE_TEST_DYNAMICMATRIX_HPP
#define TEST_CORE_TEST_DYNAMICMATRIX_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;
using DynamicMatrixUnary = to_gtest_types<DynamicMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DynamicMatrixUnaryTest : public ::testing::Test {
 public:
  using type      = UnaryContainer;
  using device    = typename UnaryContainer::type;
  using host      = typename UnaryContainer::type::HostMirror;
  using IndexType = typename device::index_type;
  using ValueType = typename device::value_type;
  using DevLayout = typename device::array_layout;
  using DevSpace  = typename device::execution_space;

  using CooDev = Morpheus::CooMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CooHost = typename CooDev::HostMirror;

  using CsrDev = Morpheus::CsrMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CsrHost = typename CsrDev::HostMirror;

  DynamicMatrixUnaryTest() : nrows(3), ncols(3), nnnz(4) {}

  void SetUp() override { switch_coo(); }

  void switch_coo() {
    Aref_coo_h.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_COO_SIZES(Aref_coo_h, this->nrows, this->ncols, this->nnnz);
    build_coomatrix(Aref_coo_h);

    Aref_coo.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_COO_SIZES(Aref_coo, this->nrows, this->ncols, this->nnnz);
    Morpheus::copy(Aref_coo_h, Aref_coo);

    Aref_dyn_h = Aref_coo_h;
    CHECK_DYNAMIC_SIZES(Aref_dyn_h, 3, 3, 4, 0);

    Aref_dyn = Aref_coo;
    CHECK_DYNAMIC_SIZES(Aref_dyn, 3, 3, 4, 0);
  }

  void switch_csr() {
    Aref_csr_h.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_CSR_SIZES(Aref_csr_h, this->nrows, this->ncols, this->nnnz);
    build_csrmatrix(Aref_csr_h);

    Aref_csr.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_CSR_SIZES(Aref_csr, this->nrows, this->ncols, this->nnnz);
    Morpheus::copy(Aref_csr_h, Aref_csr);

    Aref_dyn_h = Aref_csr_h;
    CHECK_DYNAMIC_SIZES(Aref_dyn_h, 3, 3, 4, 1);

    Aref_dyn = Aref_csr;
    CHECK_DYNAMIC_SIZES(Aref_dyn, 3, 3, 4, 1);
  }

  IndexType nrows, ncols, nnnz;

  device Aref_dyn;
  host Aref_dyn_h;

  CooDev Aref_coo;
  CooHost Aref_coo_h;

  CsrDev Aref_csr;
  CsrHost Aref_csr_h;
};

namespace Test {

/**
 * @brief Test Suite using the Unary DynamicMatrix
 *
 */
TYPED_TEST_SUITE(DynamicMatrixUnaryTest, DynamicMatrixUnary);

/**
 * @brief Testing default construction of DynamicMatrix container
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_DYNAMIC_EMPTY(A);

  HostMatrix Ah;
  CHECK_DYNAMIC_EMPTY(Ah);
}

/**
 * @brief Testing the enum value of the active type currently held by the \p
 * DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, FormatEnum) {
  using Matrix          = typename TestFixture::device;
  using value_type      = typename Matrix::value_type;
  using index_type      = typename Matrix::index_type;
  using array_layout    = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout,
                                          execution_space>::type();
  EXPECT_EQ(Morpheus::COO_FORMAT, A.format_enum());

  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(Morpheus::DIA_FORMAT, A.format_enum());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(Morpheus::CSR_FORMAT, A.format_enum());
}

/**
 * @brief Testing the enum value of the active type currently held by the \p
 * DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActiveEnum) {
  using Matrix          = typename TestFixture::device;
  using value_type      = typename Matrix::value_type;
  using index_type      = typename Matrix::index_type;
  using array_layout    = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout,
                                          execution_space>::type();
  EXPECT_EQ(Morpheus::COO_FORMAT, A.active_enum());

  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(Morpheus::DIA_FORMAT, A.active_enum());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(Morpheus::CSR_FORMAT, A.active_enum());
}

/**
 * @brief Testing the index value of the active type currently held by the \p
 * DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, FormatIndex) {
  using Matrix          = typename TestFixture::device;
  using value_type      = typename Matrix::value_type;
  using index_type      = typename Matrix::index_type;
  using array_layout    = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout,
                                          execution_space>::type();
  EXPECT_EQ(0, A.format_index());

  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(2, A.format_index());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(1, A.format_index());
}

/**
 * @brief Testing the index value of the active type currently held by the \p
 * DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActiveIndex) {
  using Matrix          = typename TestFixture::device;
  using value_type      = typename Matrix::value_type;
  using index_type      = typename Matrix::index_type;
  using array_layout    = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout,
                                          execution_space>::type();
  EXPECT_EQ(0, A.active_index());

  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(2, A.active_index());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout,
                                   execution_space>::type();
  EXPECT_EQ(1, A.active_index());
}

/**
 * @brief Testing default copy assignment of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyAssignmentHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;
  using value_type = typename HostMatrix::value_type;

  HostMatrix A1_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix A2_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Bt_h = A1_h;
  VALIDATE_COO_CONTAINER(Bt_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost Ct_h = A2_h;
  VALIDATE_COO_CONTAINER(Ct_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1_h, A2_h with Csr
  this->switch_csr();
  A1_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 1);
  A2_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Dt_h = A1_h;
  VALIDATE_CSR_CONTAINER(Dt_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost Et_h = A2_h;
  VALIDATE_CSR_CONTAINER(Et_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default copy assignment of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyAssignmentDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  Matrix A1 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix A2 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev A2t = A2;
  typename TestFixture::CooHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_COO_CONTAINER(A2t_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1, A2 with Csr
  this->switch_csr();
  A1 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 1);
  A2 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev B1t = A1;
  typename TestFixture::CsrHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev B2t = A2;
  typename TestFixture::CsrHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default copy constructor of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyConstructorHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;
  using value_type = typename HostMatrix::value_type;

  HostMatrix A1_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix A2_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Bt_h = A1_h;
  VALIDATE_COO_CONTAINER(Bt_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost Ct_h = A2_h;
  VALIDATE_COO_CONTAINER(Ct_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1_h, A2_h with Csr
  this->switch_csr();
  HostMatrix B1_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(B1_h, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix B2_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(B2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost B1t_h = B1_h;
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost B2t_h = B2_h;
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default copy constructor of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyConstructorDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  Matrix A1(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix A2(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev A2t = A2;
  typename TestFixture::CooHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_COO_CONTAINER(A2t_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1, A2 with Csr
  this->switch_csr();
  Matrix B1(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(B1, this->nrows, this->ncols, this->nnnz, 1);
  Matrix B2(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(B2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev B1t = B1;
  typename TestFixture::CsrHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev B2t = B2;
  typename TestFixture::CsrHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default move assignment of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveAssignmentHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;
  using value_type = typename HostMatrix::value_type;

  HostMatrix A1_h = std::move(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix A2_h = std::move(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Bt_h = A1_h;
  VALIDATE_COO_CONTAINER(Bt_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost Ct_h = A2_h;
  VALIDATE_COO_CONTAINER(Ct_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1_h, A2_h with Csr
  this->switch_csr();
  A1_h = std::move(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 1);
  A2_h = std::move(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Dt_h = A1_h;
  VALIDATE_CSR_CONTAINER(Dt_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost Et_h = A2_h;
  VALIDATE_CSR_CONTAINER(Et_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default move assignment of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveAssignmentDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  Matrix A1 = std::move(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix A2 = std::move(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev A2t = A2;
  typename TestFixture::CooHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_COO_CONTAINER(A2t_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1, A2 with Csr
  this->switch_csr();
  A1 = std::move(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 1);
  A2 = std::move(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev B1t = A1;
  typename TestFixture::CsrHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev B2t = A2;
  typename TestFixture::CsrHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default move construction of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveConstructorHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;
  using value_type = typename HostMatrix::value_type;

  HostMatrix A1_h(std::move(this->Aref_dyn_h));
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix A2_h(std::move(this->Aref_dyn_h));
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Bt_h = A1_h;
  VALIDATE_COO_CONTAINER(Bt_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost Ct_h = A2_h;
  VALIDATE_COO_CONTAINER(Ct_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1_h, A2_h with Csr
  this->switch_csr();
  HostMatrix B1_h(std::move(this->Aref_dyn_h));
  CHECK_DYNAMIC_SIZES(B1_h, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix B2_h(std::move(this->Aref_dyn_h));
  CHECK_DYNAMIC_SIZES(B2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost B1t_h = B1_h;
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost B2t_h = B2_h;
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing default move construction of DynamicMatrix container from
 * another DynamicMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveConstructorDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  Matrix A1(std::move(this->Aref_dyn));
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix A2(std::move(this->Aref_dyn));
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev A2t = A2;
  typename TestFixture::CooHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_COO_CONTAINER(A2t_h, this->Aref_coo_h, this->nnnz, index_type);

  // Test A1, A2 with Csr
  this->switch_csr();
  Matrix B1(std::move(this->Aref_dyn));
  CHECK_DYNAMIC_SIZES(B1, this->nrows, this->ncols, this->nnnz, 1);
  Matrix B2(std::move(this->Aref_dyn));
  CHECK_DYNAMIC_SIZES(B2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = (value_type)-3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev B1t = B1;
  typename TestFixture::CsrHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_CSR_CONTAINER(B1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev B2t = B2;
  typename TestFixture::CsrHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_CSR_CONTAINER(B2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active
 * states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateEnumNoChange) {
  using Matrix = typename TestFixture::device;

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  Matrix A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);

  A.activate(Morpheus::COO_FORMAT);
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active
 * states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateEnum) {
  using Matrix = typename TestFixture::device;

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  Matrix A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);

  A.activate(Morpheus::CSR_FORMAT);
  CHECK_DYNAMIC_SIZES(A, 0, 0, 0, 1);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active
 * states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateIndexNoChange) {
  using Matrix = typename TestFixture::device;

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  Matrix A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);

  A.activate(0);
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active
 * states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateIndex) {
  using Matrix = typename TestFixture::device;

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  Matrix A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);

  A.activate(1);
  CHECK_DYNAMIC_SIZES(A, 0, 0, 0, 1);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active
 * states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateLargerIndex) {
  using Matrix = typename TestFixture::device;

  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  typename TestFixture::CsrDev csr(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(csr_h, csr);

  Matrix A = csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);

  A.activate(1000);
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DYNAMICMATRIX_HPP