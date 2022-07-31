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
  using DevSpace = typename device::execution_space;
  using HostLayout = typename host::array_layout;
  using HostSpace = typename host::execution_space;

  using CooDev = Morpheus::CooMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CooHost = typename CooDev::HostMirror;

  using CsrDev  = Morpheus::CsrMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CsrHost = typename CsrDev::HostMirror;

  DynamicMatrixUnaryTest()
      : nrows(3),
        ncols(3),
        nnnz(4) {}

  void switch_coo(){
    // typename Morpheus::CooMatrix<ValueType, IndexType, HostLayout, HostSpace>::type Acoo_href(3,3,4);
    CooHost Acoo_href(3,3,4);
    build_coomatrix(Acoo_href);
    Ahref = Acoo_href;

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  void switch_csr(){
    // typename Morpheus::CooMatrix<ValueType, IndexType, HostLayout, HostSpace>::type Acoo_href(3,3,4);
    CsrHost Acsr_href(3,3,4);
    build_csrmatrix(Acsr_href);
    Ahref = Acsr_href;

    // Send Matrix to device
    Morpheus::copy(Ahref, Aref);
  }

  IndexType nrows, ncols, nnnz;
  device Aref;
  host Ahref;
};

namespace Test {

/**
 * @brief Test Suite using the Unary DynamicMatrix
 *
 */
TYPED_TEST_CASE(DynamicMatrixUnaryTest, DynamicMatrixUnary);

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
 * @brief Testing the enum value of the active type currently held by the \p DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  using array_layout = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::COO_FORMAT, A.format_enum());
  
  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::DIA_FORMAT, A.format_enum());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::CSR_FORMAT, A.format_enum());
}

/**
 * @brief Testing the enum value of the active type currently held by the \p DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActiveEnum) {
  using Matrix = typename TestFixture::device;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  using array_layout = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::COO_FORMAT, A.active_enum());
  
  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::DIA_FORMAT, A.active_enum());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(Morpheus::CSR_FORMAT, A.active_enum());
}

/**
 * @brief Testing the index value of the active type currently held by the \p DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  using array_layout = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(0, A.format_index());
  
  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(2, A.format_index());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(1, A.format_index());
}

/**
 * @brief Testing the index value of the active type currently held by the \p DynamicMatrix.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActiveIndex) {
  using Matrix = typename TestFixture::device;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  using array_layout = typename Matrix::array_layout;
  using execution_space = typename Matrix::execution_space;
  Matrix A = typename Morpheus::CooMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(0, A.active_index());
  
  A = typename Morpheus::DiaMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(2, A.active_index());

  A = typename Morpheus::CsrMatrix<value_type, index_type, array_layout, execution_space>::type();
  EXPECT_EQ(1, A.active_index());
}

/**
 * @brief Testing default copy assignment of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyAssignmentHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;

  HostMatrix Ah, Bh, Ch, Dh;
  CHECK_DYNAMIC_EMPTY(Ah);
  CHECK_DYNAMIC_EMPTY(Bh);
  CHECK_DYNAMIC_EMPTY(Ch);
  CHECK_DYNAMIC_EMPTY(Dh);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  Ah = coo_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 0);
  Bh = coo_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 0);

  Ch = Ah;
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 0);
  Dh = Bh;
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Ct_coo_h = Ch, Dt_coo_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo_h, coo_h, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo_h, coo_h, this->nnnz, index_type);

  // Test Ah, Bh with Csr
  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  Ah = csr_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 1);
  Bh = csr_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 1);

  Ch = Ah;
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 1);
  Dh = Bh;
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 1);

  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Ct_csr_h = Ch, Dt_csr_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Ct_csr_h, csr_h, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Dt_csr_h, csr_h, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing default copy assignment of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyAssignmentDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;

  Matrix A, B, C, D;
  CHECK_DYNAMIC_EMPTY(A);
  CHECK_DYNAMIC_EMPTY(B);
  CHECK_DYNAMIC_EMPTY(C);
  CHECK_DYNAMIC_EMPTY(D);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
  B = coo;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 0);

  C = A;
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 0);
  D = B;
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;
  Morpheus::copy(coo_h, coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev Ct_coo = C, Dt_coo = D;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo, coo, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo, coo, this->nnnz, index_type);

  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  typename TestFixture::CsrDev csr(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(csr_h, csr);

  A = csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);
  B = csr;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 1);

  C = A;
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 1);
  D = B;
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;
  Morpheus::copy(csr_h, csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev Ct_csr = C, Dt_csr = D;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Ct_csr, csr, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Dt_csr, csr, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing default copy constructor of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyConstructorHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;

  HostMatrix Ah, Bh;
  CHECK_DYNAMIC_EMPTY(Ah);
  CHECK_DYNAMIC_EMPTY(Bh);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  Ah = coo_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 0);
  Bh = coo_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 0);

  HostMatrix Ch(Ah);
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix Dh(Bh);
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Ct_coo_h = Ch, Dt_coo_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo_h, coo_h, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo_h, coo_h, this->nnnz, index_type);

  // Test Ah, Bh with Csr
  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  Ah = csr_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 1);
  Bh = csr_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 1);

  HostMatrix Eh(Ah);
  CHECK_DYNAMIC_SIZES(Eh, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix Fh(Bh);
  CHECK_DYNAMIC_SIZES(Fh, this->nrows, this->ncols, this->nnnz, 1);

  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Et_csr_h = Eh, Ft_csr_h = Fh;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Et_csr_h, csr_h, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Ft_csr_h, csr_h, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing default copy constructor of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultCopyConstructorDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;

  Matrix A, B;
  CHECK_DYNAMIC_EMPTY(A);
  CHECK_DYNAMIC_EMPTY(B);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
  B = coo;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 0);

  Matrix C(A);
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 0);
  Matrix D(B);
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;
  Morpheus::copy(coo_h, coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev Ct_coo = C, Dt_coo = D;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo, coo, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo, coo, this->nnnz, index_type);

  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  typename TestFixture::CsrDev csr(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(csr_h, csr);

  A = csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);
  B = csr;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 1);

  Matrix E(A);
  CHECK_DYNAMIC_SIZES(E, this->nrows, this->ncols, this->nnnz, 1);
  Matrix F(B);
  CHECK_DYNAMIC_SIZES(F, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;
  Morpheus::copy(csr_h, csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev Et_csr = E, Ft_csr = F;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Et_csr, csr, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Ft_csr, csr, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing default move assignment of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveAssignmentHost) {
  using HostMatrix = typename TestFixture::host;
  using index_type = typename HostMatrix::index_type;

  HostMatrix Ah, Bh, Ch, Dh;
  CHECK_DYNAMIC_EMPTY(Ah);
  CHECK_DYNAMIC_EMPTY(Bh);
  CHECK_DYNAMIC_EMPTY(Ch);
  CHECK_DYNAMIC_EMPTY(Dh);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  Ah = coo_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 0);
  Bh = coo_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 0);

  Ch = std::move(Ah);
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 0);
  Dh = std::move(Bh);
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Ct_coo_h = Ch, Dt_coo_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo_h, coo_h, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo_h, coo_h, this->nnnz, index_type);

  // Test Ah, Bh with Csr
  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  Ah = csr_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 1);
  Bh = csr_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 1);

  Ch = std::move(Ah);
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 1);
  Dh = std::move(Bh);
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 1);

  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Ct_csr_h = Ch, Dt_csr_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Ct_csr_h, csr_h, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Dt_csr_h, csr_h, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing default move assignment of DynamicMatrix container from another
 * DynamicMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, DefaultMoveAssignmentDevice) {
  using Matrix     = typename TestFixture::device;
  using index_type = typename Matrix::index_type;

  Matrix A, B, C, D;
  CHECK_DYNAMIC_EMPTY(A);
  CHECK_DYNAMIC_EMPTY(B);
  CHECK_DYNAMIC_EMPTY(C);
  CHECK_DYNAMIC_EMPTY(D);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
  B = coo;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 0);

  C = std::move(A);
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 0);
  D = std::move(B);
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;
  Morpheus::copy(coo_h, coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev Ct_coo = C, Dt_coo = D;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo, coo, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo, coo, this->nnnz, index_type);

  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  typename TestFixture::CsrDev csr(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(csr_h, csr);

  A = csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);
  B = csr;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 1);

  C = std::move(A);
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 1);
  D = std::move(B);
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;
  Morpheus::copy(csr_h, csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev Ct_csr = C, Dt_csr = D;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Ct_csr, csr, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Dt_csr, csr, this->nrows, this->nnnz, index_type);
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

  HostMatrix Ah, Bh;
  CHECK_DYNAMIC_EMPTY(Ah);
  CHECK_DYNAMIC_EMPTY(Bh);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  Ah = coo_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 0);
  Bh = coo_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 0);

  HostMatrix Ch(std::move(Ah));
  CHECK_DYNAMIC_SIZES(Ch, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix Dh(std::move(Bh));
  CHECK_DYNAMIC_SIZES(Dh, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost Ct_coo_h = Ch, Dt_coo_h = Dh;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo_h, coo_h, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo_h, coo_h, this->nnnz, index_type);

  // Test Ah, Bh with Csr
  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  Ah = csr_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz, 1);
  Bh = csr_h;
  CHECK_DYNAMIC_SIZES(Bh, this->nrows, this->ncols, this->nnnz, 1);

  HostMatrix Eh(std::move(Ah));
  CHECK_DYNAMIC_SIZES(Eh, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix Fh(std::move(Bh));
  CHECK_DYNAMIC_SIZES(Fh, this->nrows, this->ncols, this->nnnz, 1);

  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost Et_csr_h = Eh, Ft_csr_h = Fh;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Et_csr_h, csr_h, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Ft_csr_h, csr_h, this->nrows, this->nnnz, index_type);
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

  Matrix A, B;
  CHECK_DYNAMIC_EMPTY(A);
  CHECK_DYNAMIC_EMPTY(B);

  typename TestFixture::CooHost coo_h(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo_h, this->nrows, this->ncols, this->nnnz);
  build_coomatrix(coo_h);

  typename TestFixture::CooDev coo(this->nrows, this->ncols, this->nnnz);
  CHECK_COO_SIZES(coo, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(coo_h, coo);

  A = coo;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 0);
  B = coo;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 0);

  Matrix C(std::move(A));
  CHECK_DYNAMIC_SIZES(C, this->nrows, this->ncols, this->nnnz, 0);
  Matrix D(std::move(B));
  CHECK_DYNAMIC_SIZES(D, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  coo_h.row_indices(2)    = 2;
  coo_h.column_indices(1) = 1;
  coo_h.values(3)         = -3.33;
  Morpheus::copy(coo_h, coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev Ct_coo = C, Dt_coo = D;
  // Other container should reflect the same changes
  VALIDATE_COO_CONTAINER(Ct_coo, coo, this->nnnz, index_type);
  VALIDATE_COO_CONTAINER(Dt_coo, coo, this->nnnz, index_type);

  typename TestFixture::CsrHost csr_h(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr_h, this->nrows, this->ncols, this->nnnz);
  build_csrmatrix(csr_h);

  typename TestFixture::CsrDev csr(this->nrows, this->ncols, this->nnnz);
  CHECK_CSR_SIZES(csr, this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(csr_h, csr);

  A = csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz, 1);
  B = csr;
  CHECK_DYNAMIC_SIZES(B, this->nrows, this->ncols, this->nnnz, 1);

  Matrix E(std::move(A));
  CHECK_DYNAMIC_SIZES(E, this->nrows, this->ncols, this->nnnz, 1);
  Matrix F(std::move(B));
  CHECK_DYNAMIC_SIZES(F, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  csr_h.row_offsets(2)    = 2;
  csr_h.column_indices(1) = 1;
  csr_h.values(3)         = -3.33;
  Morpheus::copy(csr_h, csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev Et_csr = E, Ft_csr = F;
  // Other container should reflect the same changes
  VALIDATE_CSR_CONTAINER(Et_csr, csr, this->nrows, this->nnnz, index_type);
  VALIDATE_CSR_CONTAINER(Ft_csr, csr, this->nrows, this->nnnz, index_type);
}

/**
 * @brief Testing dynamic switching of DynamicMatrix container to other active states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateEnumNoChange) {
  using Matrix     = typename TestFixture::device;

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
 * @brief Testing dynamic switching of DynamicMatrix container to other active states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateEnum) {
  using Matrix     = typename TestFixture::device;

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
 * @brief Testing dynamic switching of DynamicMatrix container to other active states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateIndexNoChange) {
  using Matrix     = typename TestFixture::device;

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
 * @brief Testing dynamic switching of DynamicMatrix container to other active states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateIndex) {
  using Matrix     = typename TestFixture::device;

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
 * @brief Testing dynamic switching of DynamicMatrix container to other active states.
 *
 */
TYPED_TEST(DynamicMatrixUnaryTest, ActivateLargerIndex) {
  using Matrix     = typename TestFixture::device;

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