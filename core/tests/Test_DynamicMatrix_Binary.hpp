/**
 * Test_DynamicMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_DYNAMICMATRIX_BINARY_HPP
#define TEST_CORE_TEST_DYNAMICMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;

using DynamicMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DynamicMatrixTypes, DynamicMatrixTypes>::type>::type;

template <typename BinaryContainer>
class DynamicMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;
  using host2   = typename type2::type::HostMirror;

  using IndexType1 = typename device1::index_type;
  using ValueType1 = typename device1::value_type;
  using DevLayout1 = typename device1::array_layout;
  using DevSpace1  = typename device1::backend;

  using IndexType2 = typename device2::index_type;
  using ValueType2 = typename device2::value_type;
  using DevLayout2 = typename device2::array_layout;
  using DevSpace2  = typename device2::backend;

  using CsrDev1 =
      Morpheus::CsrMatrix<ValueType1, IndexType1, DevLayout1, DevSpace1>;
  using CsrHost1 = typename CsrDev1::HostMirror;

  using CooDev1 =
      Morpheus::CooMatrix<ValueType1, IndexType1, DevLayout1, DevSpace1>;
  using CooHost1 = typename CooDev1::HostMirror;

  using CsrDev2 =
      Morpheus::CsrMatrix<ValueType2, IndexType2, DevLayout2, DevSpace2>;
  using CsrHost2 = typename CsrDev2::HostMirror;

  using CooDev2 =
      Morpheus::CooMatrix<ValueType2, IndexType2, DevLayout2, DevSpace2>;
  using CooHost2 = typename CooDev2::HostMirror;

  DynamicMatrixBinaryTest()
      : Acsr_nrows(5),
        Acsr_ncols(6),
        Acsr_nnnz(9),
        Acoo_nrows(4),
        Acoo_ncols(2),
        Acoo_nnnz(5) {}

  void SetUp() override {
    build_csr1();
    build_coo2();
  }

  void build_coo2() {
    using value_type = ValueType2;
    Aref_coo_h.resize(this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
    CHECK_COO_SIZES(Aref_coo_h, this->Acoo_nrows, this->Acoo_ncols,
                    this->Acoo_nnnz);

    Aref_coo.resize(this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
    CHECK_COO_SIZES(Aref_coo, this->Acoo_nrows, this->Acoo_ncols,
                    this->Acoo_nnnz);

    Aref_dyn2_h = Aref_coo_h;
    CHECK_DYNAMIC_SIZES(Aref_dyn2_h, this->Acoo_nrows, this->Acoo_ncols,
                        this->Acoo_nnnz, Morpheus::COO_FORMAT);

    Aref_dyn2 = Aref_coo;
    CHECK_DYNAMIC_SIZES(Aref_dyn2, this->Acoo_nrows, this->Acoo_ncols,
                        this->Acoo_nnnz, Morpheus::COO_FORMAT);
  }

  void build_csr1() {
    using value_type = ValueType1;
    Aref_csr_h.resize(this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);
    CHECK_CSR_SIZES(Aref_csr_h, this->Acsr_nrows, this->Acsr_ncols,
                    this->Acsr_nnnz);

    Aref_csr.resize(this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);
    CHECK_CSR_SIZES(Aref_csr, this->Acsr_nrows, this->Acsr_ncols,
                    this->Acsr_nnnz);

    Aref_dyn1_h = Aref_csr_h;
    CHECK_DYNAMIC_SIZES(Aref_dyn1_h, this->Acsr_nrows, this->Acsr_ncols,
                        this->Acsr_nnnz, Morpheus::CSR_FORMAT);

    Aref_dyn1 = Aref_csr;
    CHECK_DYNAMIC_SIZES(Aref_dyn1, this->Acsr_nrows, this->Acsr_ncols,
                        this->Acsr_nnnz, Morpheus::CSR_FORMAT);
  }

  IndexType1 Acsr_nrows, Acsr_ncols, Acsr_nnnz;
  IndexType2 Acoo_nrows, Acoo_ncols, Acoo_nnnz;

  device1 Aref_dyn1;
  host1 Aref_dyn1_h;

  device2 Aref_dyn2;
  host2 Aref_dyn2_h;

  CsrDev1 Aref_csr;
  CsrHost1 Aref_csr_h;

  CooDev2 Aref_coo;
  CooHost2 Aref_coo_h;
};

// resize
// allocate
namespace Test {
/**
 * @brief Test Suite using the Compatible Binary DynamicMatrix pairs
 *
 */
TYPED_TEST_CASE(DynamicMatrixBinaryTest, DynamicMatrixBinary);

TYPED_TEST(DynamicMatrixBinaryTest, ResizeFromDynamicMatrixHost) {
  using HostMatrix1 = typename TestFixture::host1;
  using HostMatrix2 = typename TestFixture::host2;

  using CooHostMatrix1 = typename TestFixture::CooHost1;
  using CsrHostMatrix2 = typename TestFixture::CsrHost2;

  HostMatrix1 A1_h(this->Aref_csr_h);
  HostMatrix2 A2_h(this->Aref_coo_h), A3_h(this->Aref_coo_h);

  CHECK_DYNAMIC_SIZES(A1_h, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);
  CHECK_DYNAMIC_SIZES(A2_h, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  A2_h.resize(A1_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);

  CsrHostMatrix2 A2csr = A2_h;
  CHECK_CSR_SIZES(A2csr, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);

  A1_h.resize(A3_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  CooHostMatrix1 A1csr = A1_h;
  CHECK_COO_SIZES(A1csr, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
}

TYPED_TEST(DynamicMatrixBinaryTest, ResizeFromDynamicMatrixDevice) {
  using Matrix1 = typename TestFixture::device1;
  using Matrix2 = typename TestFixture::device2;

  using CooMatrix1 = typename TestFixture::CooDev1;
  using CsrMatrix2 = typename TestFixture::CsrDev2;

  Matrix1 A1(this->Aref_csr);
  Matrix2 A2(this->Aref_coo), A3(this->Aref_coo);

  CHECK_DYNAMIC_SIZES(A1, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);
  CHECK_DYNAMIC_SIZES(A2, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  A2.resize(A1);
  CHECK_DYNAMIC_SIZES(A2, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);

  CsrMatrix2 A2csr = A2;
  CHECK_CSR_SIZES(A2csr, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);

  A1.resize(A3);
  CHECK_DYNAMIC_SIZES(A1, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  CooMatrix1 A1csr = A1;
  CHECK_COO_SIZES(A1csr, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
}

TYPED_TEST(DynamicMatrixBinaryTest, AllocateFromDynamicMatrixHost) {
  using HostMatrix1 = typename TestFixture::host1;
  using HostMatrix2 = typename TestFixture::host2;

  using CooHostMatrix1 = typename TestFixture::CooHost1;
  using CsrHostMatrix2 = typename TestFixture::CsrHost2;

  HostMatrix1 A1_h(this->Aref_csr_h);
  HostMatrix2 A2_h(this->Aref_coo_h), A3_h(this->Aref_coo_h);

  CHECK_DYNAMIC_SIZES(A1_h, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);
  CHECK_DYNAMIC_SIZES(A2_h, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  A2_h.resize(A1_h);
  CHECK_DYNAMIC_SIZES(A2_h, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);

  CsrHostMatrix2 A2csr = A2_h;
  CHECK_CSR_SIZES(A2csr, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);

  A1_h.resize(A3_h);
  CHECK_DYNAMIC_SIZES(A1_h, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  CooHostMatrix1 A1coo = A1_h;
  CHECK_COO_SIZES(A1coo, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
}

TYPED_TEST(DynamicMatrixBinaryTest, AllocateFromDynamicMatrixDevice) {
  using Matrix1 = typename TestFixture::device1;
  using Matrix2 = typename TestFixture::device2;

  using CooMatrix1 = typename TestFixture::CooDev1;
  using CsrMatrix2 = typename TestFixture::CsrDev2;

  Matrix1 A1(this->Aref_csr);
  Matrix2 A2(this->Aref_coo), A3(this->Aref_coo);

  CHECK_DYNAMIC_SIZES(A1, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);
  CHECK_DYNAMIC_SIZES(A2, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  A2.allocate(A1);
  CHECK_DYNAMIC_SIZES(A2, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz,
                      Morpheus::CSR_FORMAT);

  CsrMatrix2 A2csr = A2;
  CHECK_CSR_SIZES(A2csr, this->Acsr_nrows, this->Acsr_ncols, this->Acsr_nnnz);

  A1.allocate(A3);
  CHECK_DYNAMIC_SIZES(A1, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz,
                      Morpheus::COO_FORMAT);

  CooMatrix1 A1coo = A1;
  CHECK_COO_SIZES(A1coo, this->Acoo_nrows, this->Acoo_ncols, this->Acoo_nnnz);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DYNAMICMATRIX_BINARY_HPP