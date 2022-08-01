/**
 * Test_DynamicMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

using DynamicMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DynamicMatrix<double>, types::compatible_types_set>::type;

using CompatibleDynamicMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DynamicMatrixCompatibleTypes,
        DynamicMatrixCompatibleTypes>::type>::type;

// Used for testing unary operations for same type container
template <typename BinaryContainer>
class CompatibleDynamicMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;
  using host2   = typename type2::type::HostMirror;

  using IndexType = typename device1::index_type;
  using ValueType = typename device1::value_type;
  using DevLayout = typename device1::array_layout;
  using DevSpace  = typename device1::execution_space;

  using CooDev = Morpheus::CooMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CooHost = typename CooDev::HostMirror;

  using CsrDev = Morpheus::CsrMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CsrHost = typename CsrDev::HostMirror;

  CompatibleDynamicMatrixBinaryTest() : nrows(3), ncols(3), nnnz(4) {}

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

  device1 Aref_dyn;
  host1 Aref_dyn_h;

  CooDev Aref_coo;
  CooHost Aref_coo_h;

  CsrDev Aref_csr;
  CsrHost Aref_csr_h;
};

namespace Test {
/**
 * @brief Test Suite using the Compatible Binary DynamicMatrix pairs
 *
 */
TYPED_TEST_CASE(CompatibleDynamicMatrixBinaryTest,
                CompatibleDynamicMatrixBinary);

TYPED_TEST(CompatibleDynamicMatrixBinaryTest,
           ConstructionFromDynamicMatrixHost) {
  using HostMatrix1 = typename TestFixture::host1;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename HostMatrix2::index_type;

  // Two containers aliasing the same matrix - same properties
  HostMatrix1 A1_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix1 A2_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Construct compatible containers and check if aliasing is still maintained
  HostMatrix2 B1_h(A1_h);
  CHECK_DYNAMIC_SIZES(B1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix2 B2_h(A2_h);
  CHECK_DYNAMIC_SIZES(B2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost B1t_h = B1_h;
  VALIDATE_COO_CONTAINER(B1t_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost B2t_h = B2_h;
  VALIDATE_COO_CONTAINER(B2t_h, this->Aref_coo_h, this->nnnz, index_type);

  this->switch_csr();
  A1_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 1);
  A2_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Construct compatible containers and check if aliasing is still maintained
  HostMatrix2 C1_h(A1_h);
  CHECK_DYNAMIC_SIZES(C1_h, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix2 C2_h(A2_h);
  CHECK_DYNAMIC_SIZES(C2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost C1t_h = C1_h;
  VALIDATE_CSR_CONTAINER(C1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost C2t_h = C2_h;
  VALIDATE_CSR_CONTAINER(C2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

TYPED_TEST(CompatibleDynamicMatrixBinaryTest,
           ConstructionFromDynamicMatrixDevice) {
  using Matrix1    = typename TestFixture::device1;
  using Matrix2    = typename TestFixture::device2;
  using index_type = typename Matrix2::index_type;

  // Two containers aliasing the same matrix - same properties
  Matrix1 A1(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix1 A2(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Construct compatible containers and check if aliasing is still maintained
  Matrix2 B1(A1);
  CHECK_DYNAMIC_SIZES(B1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix2 B2(A2);
  CHECK_DYNAMIC_SIZES(B2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev B1t = B1;
  typename TestFixture::CooHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_COO_CONTAINER(B1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev B2t = B2;
  typename TestFixture::CooHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_COO_CONTAINER(B2t_h, this->Aref_coo_h, this->nnnz, index_type);

  this->switch_csr();
  A1 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 1);
  A2 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 1);

  // Construct compatible containers and check if aliasing is still maintained
  Matrix2 C1(A1);
  CHECK_DYNAMIC_SIZES(C1, this->nrows, this->ncols, this->nnnz, 1);
  Matrix2 C2(A2);
  CHECK_DYNAMIC_SIZES(C2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev C1t = C1;
  typename TestFixture::CsrHost C1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(C1t, C1t_h);
  VALIDATE_CSR_CONTAINER(C1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev C2t = C2;
  typename TestFixture::CsrHost C2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(C2t, C2t_h);
  VALIDATE_CSR_CONTAINER(C2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

TYPED_TEST(CompatibleDynamicMatrixBinaryTest,
           CopyAssignmentFromDynamicMatrixHost) {
  using HostMatrix1 = typename TestFixture::host1;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename HostMatrix2::index_type;

  // Two containers aliasing the same matrix - same properties
  HostMatrix1 A1_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix1 A2_h(this->Aref_dyn_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Construct compatible containers and check if aliasing is still maintained
  HostMatrix2 B1_h = A1_h;
  CHECK_DYNAMIC_SIZES(B1_h, this->nrows, this->ncols, this->nnnz, 0);
  HostMatrix2 B2_h = A2_h;
  CHECK_DYNAMIC_SIZES(B2_h, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost B1t_h = B1_h;
  VALIDATE_COO_CONTAINER(B1t_h, this->Aref_coo_h, this->nnnz, index_type);
  typename TestFixture::CooHost B2t_h = B2_h;
  VALIDATE_COO_CONTAINER(B2t_h, this->Aref_coo_h, this->nnnz, index_type);

  this->switch_csr();
  A1_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz, 1);
  A2_h = this->Aref_dyn_h;
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Construct compatible containers and check if aliasing is still maintained
  HostMatrix2 C1_h = A1_h;
  CHECK_DYNAMIC_SIZES(C1_h, this->nrows, this->ncols, this->nnnz, 1);
  HostMatrix2 C2_h = A2_h;
  CHECK_DYNAMIC_SIZES(C2_h, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrHost C1t_h = C1_h;
  VALIDATE_CSR_CONTAINER(C1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
  typename TestFixture::CsrHost C2t_h = C2_h;
  VALIDATE_CSR_CONTAINER(C2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

TYPED_TEST(CompatibleDynamicMatrixBinaryTest,
           CopyAssignmentFromDynamicMatrixDevice) {
  using Matrix1    = typename TestFixture::device1;
  using Matrix2    = typename TestFixture::device2;
  using index_type = typename Matrix2::index_type;

  // Two containers aliasing the same matrix - same properties
  Matrix1 A1(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix1 A2(this->Aref_dyn);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 0);

  // Construct compatible containers and check if aliasing is still maintained
  Matrix2 B1 = A1;
  CHECK_DYNAMIC_SIZES(B1, this->nrows, this->ncols, this->nnnz, 0);
  Matrix2 B2 = A2;
  CHECK_DYNAMIC_SIZES(B2, this->nrows, this->ncols, this->nnnz, 0);

  // Change values in one container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev B1t = B1;
  typename TestFixture::CooHost B1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B1t, B1t_h);
  VALIDATE_COO_CONTAINER(B1t_h, this->Aref_coo_h, this->nnnz, index_type);

  typename TestFixture::CooDev B2t = B2;
  typename TestFixture::CooHost B2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(B2t, B2t_h);
  VALIDATE_COO_CONTAINER(B2t_h, this->Aref_coo_h, this->nnnz, index_type);

  this->switch_csr();
  A1 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz, 1);
  A2 = this->Aref_dyn;
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz, 1);

  // Construct compatible containers and check if aliasing is still maintained
  Matrix2 C1 = A1;
  CHECK_DYNAMIC_SIZES(C1, this->nrows, this->ncols, this->nnnz, 1);
  Matrix2 C2 = A2;
  CHECK_DYNAMIC_SIZES(C2, this->nrows, this->ncols, this->nnnz, 1);

  // Change values in one container
  this->Aref_csr_h.row_offsets(2)    = 2;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CsrDev C1t = C1;
  typename TestFixture::CsrHost C1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(C1t, C1t_h);
  VALIDATE_CSR_CONTAINER(C1t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);

  typename TestFixture::CsrDev C2t = C2;
  typename TestFixture::CsrHost C2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(C2t, C2t_h);
  VALIDATE_CSR_CONTAINER(C2t_h, this->Aref_csr_h, this->nrows, this->nnnz,
                         index_type);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLEBINARY_HPP