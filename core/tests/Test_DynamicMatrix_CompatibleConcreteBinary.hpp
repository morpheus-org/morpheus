/**
 * Test_DynamicMatrix_CompatibleConcreteBinary.hpp
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

#ifndef TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLECONCRETEBINARY_HPP
#define TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLECONCRETEBINARY_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_Definitions.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;
using DynamicMatrixUnary = to_gtest_types<DynamicMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class CompatibleConcreteBinaryTest : public ::testing::Test {
 public:
  using type      = UnaryContainer;
  using device    = typename UnaryContainer::type;
  using host      = typename UnaryContainer::type::HostMirror;
  using SizeType  = typename device::size_type;
  using IndexType = typename device::index_type;
  using ValueType = typename device::value_type;
  using DevLayout = typename device::array_layout;
  using DevSpace  = typename device::backend;

  using CooDev = Morpheus::CooMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CooHost = typename CooDev::HostMirror;

  using CsrDev = Morpheus::CsrMatrix<ValueType, IndexType, DevLayout, DevSpace>;
  using CsrHost = typename CsrDev::HostMirror;

  CompatibleConcreteBinaryTest()
      : nrows(SMALL_MATRIX_NROWS),
        ncols(SMALL_MATRIX_NCOLS),
        nnnz(SMALL_MATRIX_NNZ) {}

  void SetUp() override {
    build_coo();
    build_csr();
  }

  void build_coo() {
    Aref_coo_h.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_COO_SIZES(Aref_coo_h, this->nrows, this->ncols, this->nnnz);
    Morpheus::Test::build_small_container(Aref_coo_h);

    Aref_coo.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_COO_SIZES(Aref_coo, this->nrows, this->ncols, this->nnnz);
    Morpheus::copy(Aref_coo_h, Aref_coo);
  }

  void build_csr() {
    Aref_csr_h.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_CSR_SIZES(Aref_csr_h, this->nrows, this->ncols, this->nnnz);
    Morpheus::Test::build_small_container(Aref_csr_h);

    Aref_csr.resize(this->nrows, this->ncols, this->nnnz);
    CHECK_CSR_SIZES(Aref_csr, this->nrows, this->ncols, this->nnnz);
    Morpheus::copy(Aref_csr_h, Aref_csr);
  }

  SizeType nrows, ncols, nnnz;

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
TYPED_TEST_SUITE(CompatibleConcreteBinaryTest, DynamicMatrixUnary);

TYPED_TEST(CompatibleConcreteBinaryTest, ConstructionFromConcreteHost) {
  using HostMatrix = typename TestFixture::host;

  // Two containers aliasing the same matrix - same properties
  HostMatrix A1_h(this->Aref_coo_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz,
                      Morpheus::COO_FORMAT);
  HostMatrix A2_h(this->Aref_csr_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  // Change values in concrete container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;

  this->Aref_csr_h.row_offsets(2)    = 6;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost A1t_h = A1_h;
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz);
  typename TestFixture::CsrHost A2t_h = A2_h;
  VALIDATE_CSR_CONTAINER(A2t_h, this->Aref_csr_h, this->nrows, this->nnnz);
}

TYPED_TEST(CompatibleConcreteBinaryTest, ConstructionFromConcreteDevice) {
  using Matrix = typename TestFixture::device;

  // Two containers aliasing the same matrix - same properties
  Matrix A1(this->Aref_coo);
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz,
                      Morpheus::COO_FORMAT);
  Matrix A2(this->Aref_csr);
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  // Change values in concrete container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  this->Aref_csr_h.row_offsets(2)    = 6;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz);

  typename TestFixture::CsrDev A2t = A2;
  typename TestFixture::CsrHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_CSR_CONTAINER(A2t_h, this->Aref_csr_h, this->nrows, this->nnnz);
}

TYPED_TEST(CompatibleConcreteBinaryTest, CopyAssignmentFromConcreteHost) {
  using HostMatrix = typename TestFixture::host;

  // Two containers aliasing the same matrix - same properties
  HostMatrix A1_h = this->Aref_coo_h;
  CHECK_DYNAMIC_SIZES(A1_h, this->nrows, this->ncols, this->nnnz,
                      Morpheus::COO_FORMAT);
  HostMatrix A2_h = this->Aref_csr_h;
  CHECK_DYNAMIC_SIZES(A2_h, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  // Change values in concrete container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;

  this->Aref_csr_h.row_offsets(2)    = 6;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooHost A1t_h = A1_h;
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz);
  typename TestFixture::CsrHost A2t_h = A2_h;
  VALIDATE_CSR_CONTAINER(A2t_h, this->Aref_csr_h, this->nrows, this->nnnz);
}

TYPED_TEST(CompatibleConcreteBinaryTest, CopyAssignmentFromConcreteDevice) {
  using Matrix = typename TestFixture::device;

  // Two containers aliasing the same matrix - same properties
  Matrix A1 = this->Aref_coo;
  CHECK_DYNAMIC_SIZES(A1, this->nrows, this->ncols, this->nnnz,
                      Morpheus::COO_FORMAT);
  Matrix A2 = this->Aref_csr;
  CHECK_DYNAMIC_SIZES(A2, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  // Change values in concrete container
  this->Aref_coo_h.row_indices(2)    = 2;
  this->Aref_coo_h.column_indices(1) = 1;
  this->Aref_coo_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_coo_h, this->Aref_coo);

  this->Aref_csr_h.row_offsets(2)    = 6;
  this->Aref_csr_h.column_indices(1) = 1;
  this->Aref_csr_h.values(3)         = -3.33;
  Morpheus::copy(this->Aref_csr_h, this->Aref_csr);

  // Extract active state of both Dynamic containers to check
  typename TestFixture::CooDev A1t = A1;
  typename TestFixture::CooHost A1t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A1t, A1t_h);
  VALIDATE_COO_CONTAINER(A1t_h, this->Aref_coo_h, this->nnnz);

  typename TestFixture::CsrDev A2t = A2;
  typename TestFixture::CsrHost A2t_h(this->nrows, this->ncols, this->nnnz);
  Morpheus::copy(A2t, A2t_h);
  VALIDATE_CSR_CONTAINER(A2t_h, this->Aref_csr_h, this->nrows, this->nnnz);
}

TYPED_TEST(CompatibleConcreteBinaryTest, ResizeFromConcreteHost) {
  using HostMatrix    = typename TestFixture::host;
  using CooHostMatrix = typename TestFixture::CooHost;
  using size_type     = typename HostMatrix::size_type;

  HostMatrix Ah = this->Aref_csr_h;
  CHECK_DYNAMIC_SIZES(Ah, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  const size_type coo_rows = 15, coo_cols = 10, coo_nnnz = 20;
  CooHostMatrix Acoo_h(15, 10, 20);
  Ah.resize(Acoo_h);
  CHECK_DYNAMIC_SIZES(Ah, coo_rows, coo_cols, coo_nnnz, Morpheus::COO_FORMAT);

  CooHostMatrix Afrom_dyn_h = Ah;
  CHECK_COO_SIZES(Afrom_dyn_h, coo_rows, coo_cols, coo_nnnz);
}

TYPED_TEST(CompatibleConcreteBinaryTest, ResizeFromConcreteDevice) {
  using Matrix       = typename TestFixture::device;
  using CooDevMatrix = typename TestFixture::CooDev;
  using size_type    = typename Matrix::size_type;

  Matrix A = this->Aref_csr;
  CHECK_DYNAMIC_SIZES(A, this->nrows, this->ncols, this->nnnz,
                      Morpheus::CSR_FORMAT);

  const size_type coo_rows = 15, coo_cols = 10, coo_nnnz = 20;
  CooDevMatrix Acoo(15, 10, 20);
  A.resize(Acoo);
  CHECK_DYNAMIC_SIZES(A, coo_rows, coo_cols, coo_nnnz, Morpheus::COO_FORMAT);

  CooDevMatrix Afrom_dyn = A;
  CHECK_COO_SIZES(Afrom_dyn, coo_rows, coo_cols, coo_nnnz);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DYNAMICMATRIX_COMPATIBLECONCRETEBINARY_HPP