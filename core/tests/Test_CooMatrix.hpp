/**
 * Test_CooMatrix.hpp
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

#ifndef TEST_CORE_TEST_COOMATRIX_HPP
#define TEST_CORE_TEST_COOMATRIX_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::test_types_set>::type;
using CooMatrixUnary = to_gtest_types<CooMatrixTypes>::type;

using CooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixTypes, CooMatrixTypes>::type>::type;

using CooMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::CooMatrix<double>, types::compatible_types_set>::type;

using CompatibleCooMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        CooMatrixCompatibleTypes, CooMatrixCompatibleTypes>::type>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class CooMatrixUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;
};

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
};

// Used for testing binary operations
template <typename BinaryContainer>
class CooMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // CooMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // CooMatrix
  using host2   = typename type2::type::HostMirror;
};

namespace Test {
#define CHECK_COO_SIZES(A, num_rows, num_cols, num_nnz) \
  {                                                     \
    EXPECT_EQ(A.nrows(), num_rows);                     \
    EXPECT_EQ(A.ncols(), num_cols);                     \
    EXPECT_EQ(A.nnnz(), num_nnz);                       \
    EXPECT_EQ(A.row_indices().size(), num_nnz);         \
    EXPECT_EQ(A.column_indices().size(), num_nnz);      \
    EXPECT_EQ(A.values().size(), num_nnz);              \
  }

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
 * @brief Test Suite using the Unary CooMatrix
 *
 */
TYPED_TEST_CASE(CooMatrixUnaryTest, CooMatrixUnary);

/**
 * @brief Testing default construction of CooMatrix container
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  CHECK_COO_SIZES(A, 0, 0, 0);
  EXPECT_EQ(A.row_indices().data(), nullptr);
  EXPECT_EQ(A.column_indices().data(), nullptr);
  EXPECT_EQ(A.values().data(), nullptr);

  HostMatrix Ah;
  CHECK_COO_SIZES(Ah, 0, 0, 0);
  EXPECT_EQ(Ah.row_indices().data(), nullptr);
  EXPECT_EQ(Ah.column_indices().data(), nullptr);
  EXPECT_EQ(Ah.values().data(), nullptr);
}

/**
 * @brief Testing the enum value assigned to the container is what we expect it
 * to be i.e COO_FORMAT.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, FormatEnum) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(Morpheus::COO_FORMAT, A.format_enum());
}

/**
 * @brief Testing the format index assigned to the container is what we expect
 * it to be i.e 0.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, FormatIndex) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_EQ(0, A.format_index());
}

TYPED_TEST(CooMatrixUnaryTest, ReferenceByIndex) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), ih[n]);
    EXPECT_EQ(Ah.column_indices(n), jh[n]);
    EXPECT_EQ(Ah.values(n), vh[n]);

    EXPECT_EQ(Ah.crow_indices(n), ih[n]);
    EXPECT_EQ(Ah.ccolumn_indices(n), jh[n]);
    EXPECT_EQ(Ah.cvalues(n), vh[n]);
  }
}

TYPED_TEST(CooMatrixUnaryTest, Reference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  index_array_type rind = A.row_indices();
  index_array_type cind = A.column_indices();
  value_array_type vals = A.values();

  host_index_array_type rind_h(nnnz, 0);
  Morpheus::copy(rind, rind_h);
  host_index_array_type cind_h(nnnz, 0);
  Morpheus::copy(cind, cind_h);
  host_value_array_type vals_h(nnnz, 0);
  Morpheus::copy(vals, vals_h);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(rind_h[n], ih[n]);
    EXPECT_EQ(cind_h[n], jh[n]);
    EXPECT_EQ(vals_h[n], vh[n]);
  }
}

TYPED_TEST(CooMatrixUnaryTest, ConstReference) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  const index_array_type rind = A.crow_indices();
  const index_array_type cind = A.ccolumn_indices();
  const value_array_type vals = A.cvalues();
  host_index_array_type rind_test(nnnz, 0);
  Morpheus::copy(rind, rind_test);
  host_index_array_type cind_test(nnnz, 0);
  Morpheus::copy(cind, cind_test);
  host_value_array_type vals_test(nnnz, 0);
  Morpheus::copy(vals, vals_test);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(rind_test[n], ih[n]);
    EXPECT_EQ(cind_test[n], jh[n]);
    EXPECT_EQ(vals_test[n], vh[n]);
  }

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  const host_index_array_type rind_h = Ah.crow_indices();
  const host_index_array_type cind_h = Ah.ccolumn_indices();
  const host_value_array_type vals_h = Ah.cvalues();

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(rind_h[n], ih[n]);
    EXPECT_EQ(cind_h[n], jh[n]);
    EXPECT_EQ(vals_h[n], vh[n]);
  }
}

/**
 * @brief Testing default copy assignment of CooMatrix container from another
 * CooMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), ih[n]);
    EXPECT_EQ(Ah.column_indices(n), jh[n]);
    EXPECT_EQ(Ah.values(n), vh[n]);
  }

  // Default copy asssignment
  HostMatrix Bh = Ah;
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ah.values(n));
  }

  // Now check device Matrix
  Matrix B = A;
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

/**
 * @brief Testing default copy constructor of CooMatrix container from another
 * CooMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), ih[n]);
    EXPECT_EQ(Ah.column_indices(n), jh[n]);
    EXPECT_EQ(Ah.values(n), vh[n]);
  }

  // Default copy asssignment
  HostMatrix Bh(Ah);
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ah.values(n));
  }

  // Now check device Matrix
  Matrix B(A);
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

/**
 * @brief Testing default move assignment of CooMatrix container from another
 * CooMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), ih[n]);
    EXPECT_EQ(Ah.column_indices(n), jh[n]);
    EXPECT_EQ(Ah.values(n), vh[n]);
  }

  // Default copy asssignment
  HostMatrix Bh = std::move(Ah);
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ah.values(n));
  }

  // Now check device Matrix
  Matrix B = std::move(A);
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

/**
 * @brief Testing default move construction of CooMatrix container from
 * another CooMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1 * 2]
  // [* * 3]
  // [* 4 *]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_array_type i(nnnz, 0), j(nnnz, 0);
  value_array_type v(nnnz, 0);
  // Construct vectors on host
  host_index_array_type ih(nnnz, 0), jh(nnnz, 0);
  host_value_array_type vh(nnnz, 0);
  // clang-format off
  ih[0] = 0; jh[0] = 0; vh[0] = 1.11;
  ih[1] = 0; jh[1] = 2; vh[1] = 2.22;
  ih[2] = 1; jh[2] = 2; vh[2] = 3.33;
  ih[3] = 2; jh[3] = 1; vh[3] = 4.44;
  // clang-format on

  // Send vectors to device
  Morpheus::copy(ih, i);
  Morpheus::copy(jh, j);
  Morpheus::copy(vh, v);

  // Build matrix from the device vectors
  Matrix A(nrows, ncols, nnnz, i, j, v);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz);

  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), ih[n]);
    EXPECT_EQ(Ah.column_indices(n), jh[n]);
    EXPECT_EQ(Ah.values(n), vh[n]);
  }

  // Default copy asssignment
  HostMatrix Bh(std::move(Ah));
  CHECK_COO_CONTAINERS(Bh, Ah);

  // Change values in one container
  Ah.row_indices(2)    = 2;
  Ah.column_indices(1) = 1;
  Ah.values(3)         = -3.33;

  // Other container should reflect the same changes
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bh.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bh.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bh.values(n), Ah.values(n));
  }

  // Now check device Matrix
  Matrix B(std::move(A));
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

// TYPED_TEST(CooMatrixUnaryTest, ConstructionFromShape) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, ConstructionFromDenseVector) { EXPECT_EQ(1,
// 0); }

// TYPED_TEST(CooMatrixUnaryTest, Resize) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, SortByRow) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, SortByRowAndColumn) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, IsSortedByRow) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixUnaryTest, IsSorted) { EXPECT_EQ(1, 0); }

// /**
//  * @brief Test Suite using the Compatible Binary CooMatrix pairs
//  *
//  */
// TYPED_TEST_CASE(CompatibleCooMatrixBinaryTest, CompatibleCooMatrixBinary);

// TYPED_TEST(CompatibleCooMatrixBinaryTest, ConstructionFromCooMatrix) {
//   EXPECT_EQ(1, 0);
// }

// TYPED_TEST(CompatibleCooMatrixBinaryTest, CopyAssignmentFromCooMatrix) {
//   EXPECT_EQ(1, 0);
// }

// /**
//  * @brief Test Suite using the Binary CooMatrix pairs
//  *
//  */
// TYPED_TEST_CASE(CooMatrixBinaryTest, CooMatrixBinary);

// TYPED_TEST(CooMatrixBinaryTest, ResizeFromCooMatrix) { EXPECT_EQ(1, 0); }

// TYPED_TEST(CooMatrixBinaryTest, AllocateFromCooMatrix) { EXPECT_EQ(1, 0); }

// /**
//  * @brief Test Suite using the Binary CooMatrix-DynamicMatrix Compatible
//  pairs
//  *
//  */
// TYPED_TEST_CASE(CompatibleCooMatrixDynamicTest, CooMatrixDynamic);

// TYPED_TEST(CompatibleCooMatrixDynamicTest, ConstructionFromDynamicMatrix) {
//   EXPECT_EQ(1, 0);
// }

// TYPED_TEST(CompatibleCooMatrixDynamicTest, CopyAssignmentFromDynamicMatrix) {
//   EXPECT_EQ(1, 0);
// }
}  // namespace Test
#endif  // TEST_CORE_TEST_COOMATRIX_HPP