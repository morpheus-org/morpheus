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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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

TYPED_TEST(CooMatrixUnaryTest, ConstructionFromShape) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  index_type nrows = 3, ncols = 3, nnnz = 4;
  HostMatrix Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), (index_type)0);
    EXPECT_EQ(Ah.column_indices(n), (index_type)0);
    EXPECT_EQ(Ah.values(n), (value_type)0);
  }
  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]

  // clang-format off
  Ah.row_indices(0) = 0; Ah.column_indices(0) = 0; Ah.values(0) = 1.11;
  Ah.row_indices(1) = 0; Ah.column_indices(1) = 2; Ah.values(1) = 2.22;
  Ah.row_indices(2) = 1; Ah.column_indices(2) = 2; Ah.values(2) = 3.33;
  Ah.row_indices(3) = 2; Ah.column_indices(3) = 1; Ah.values(3) = 4.44;
  // clang-format on

  Matrix A(nrows, ncols, nnnz);
  CHECK_COO_SIZES(A, nrows, ncols, nnnz)
  // Send vectors to device
  Morpheus::copy(Ah, A);

  HostMatrix Ah_test(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah_test, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah_test);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah_test.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Ah_test.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Ah_test.values(n), Ah.values(n));
  }
}

// /**
//  * @brief Testing construction of CooMatrix from a raw pointers
//  *
//  */
// TYPED_TEST(CooMatrixUnaryTest, ConstructionFromPointers) { EXPECT_EQ(1, 0); }

/**
 * @brief Testing construction of CooMatrix from \p DenseVector arrays.
 *
 */
TYPED_TEST(CooMatrixUnaryTest, ConstructionFromDenseVector) {
  using Matrix                = typename TestFixture::device;
  using HostMatrix            = typename TestFixture::host;
  using index_type            = typename Matrix::index_type;
  using index_array_type      = typename Matrix::index_array_type;
  using value_array_type      = typename Matrix::value_array_type;
  using host_index_array_type = typename index_array_type::HostMirror;
  using host_value_array_type = typename value_array_type::HostMirror;

  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
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

  // Build matrix from the host vectors
  HostMatrix Ah(nrows, ncols, nnnz, ih, jh, vh);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz);

  HostMatrix Ah_test(nrows, ncols, nnnz);
  Morpheus::copy(A, Ah_test);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), Ah_test.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ah_test.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ah_test.values(n));
  }

  Ah.row_indices(2) = 2;
  EXPECT_NE(Ah.row_indices(2), Ah_test.row_indices(2));
  Ah.column_indices(1) = 1;
  EXPECT_NE(Ah.column_indices(1), Ah_test.column_indices(1));
  Ah.values(0) = -1.11;
  EXPECT_NE(Ah.values(0), Ah_test.values(0));
}

TYPED_TEST(CooMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  index_type large_nrows = 500, large_ncols = 400, large_nnnz = 640;
  index_type small_nrows = 2, small_ncols = 3, small_nnnz = 2;

  HostMatrix Ahref(nrows, ncols, nnnz);
  // clang-format off
  Ahref.row_indices(0) = 0; Ahref.column_indices(0) = 0; Ahref.values(0) = 1.11;
  Ahref.row_indices(1) = 0; Ahref.column_indices(1) = 2; Ahref.values(1) = 2.22;
  Ahref.row_indices(2) = 1; Ahref.column_indices(2) = 2; Ahref.values(2) = 3.33;
  Ahref.row_indices(3) = 2; Ahref.column_indices(3) = 1; Ahref.values(3) = 4.44;
  // clang-format on

  Matrix Aref(nrows, ncols, nnnz);
  Morpheus::copy(Ahref, Aref);

  Matrix A(nrows, ncols, nnnz);
  Morpheus::copy(Ahref, A);

  // Resize to larger shape and non-zeros
  A.resize(large_nrows, large_ncols, large_nnnz);
  CHECK_COO_SIZES(A, large_nrows, large_ncols, large_nnnz);

  HostMatrix Ah(large_nrows, large_ncols, large_nnnz);
  CHECK_COO_SIZES(Ah, large_nrows, large_ncols, large_nnnz);
  Morpheus::copy(A, Ah);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), Ahref.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ahref.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref.values(n));
  }

  // Resizing to larger sizes should invoke a new allocation so changes in
  // matrix should not be reflected in reference
  Ah.row_indices(1)    = 1;
  Ah.column_indices(2) = 10;
  Ah.values(0)         = -1.11;
  Morpheus::copy(Ah, A);

  // Copy reference back to see if there are any changes
  HostMatrix Ahref_test(nrows, ncols, nnnz);
  Morpheus::copy(Ahref, Ahref_test);
  EXPECT_NE(Ah.row_indices(1), Ahref_test.row_indices(1));
  EXPECT_NE(Ah.column_indices(2), Ahref_test.column_indices(2));
  EXPECT_NE(Ah.values(0), Ahref_test.values(0));

  for (index_type n = nnnz; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.row_indices(n), 0);
    EXPECT_EQ(Ah.column_indices(n), 0);
    EXPECT_EQ(Ah.values(n), 0);
  }

  // Resize to smaller shape and non-zeros
  A.resize(small_nrows, small_ncols, small_nnnz);
  CHECK_COO_SIZES(A, small_nrows, small_ncols, small_nnnz);
  Ah.resize(small_nrows, small_ncols, small_nnnz);
  CHECK_COO_SIZES(Ah, small_nrows, small_ncols, small_nnnz);

  // Set back to normal
  Ah.row_indices(1)    = 0;
  Ah.column_indices(2) = 1;
  Ah.values(0)         = 1.11;
  Morpheus::copy(Ah, A);

  for (index_type n = 0; n < Ah.nnnz(); n++) {
    EXPECT_EQ(Ah.row_indices(n), Ahref_test.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ahref_test.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref_test.values(n));
  }
}

TYPED_TEST(CooMatrixUnaryTest, SortByRow) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_THROW(A.sort_by_row(), Morpheus::NotImplementedException);
}

TYPED_TEST(CooMatrixUnaryTest, Sort) {
  using Matrix = typename TestFixture::device;
  using space  = typename Matrix::execution_space;

  Matrix A(5, 5, 7);

  if (Morpheus::is_serial_execution_space<space>::value) {
    // clang-format off
    A.row_indices(0) = 3; A.column_indices(0) = 1; A.values(0) = 1;
    A.row_indices(1) = 4; A.column_indices(1) = 2; A.values(1) = 2;
    A.row_indices(2) = 1; A.column_indices(2) = 3; A.values(2) = 3;
    A.row_indices(3) = 2; A.column_indices(3) = 2; A.values(3) = 4;
    A.row_indices(4) = 1; A.column_indices(4) = 2; A.values(4) = 5;
    A.row_indices(5) = 0; A.column_indices(5) = 3; A.values(5) = 6;
    A.row_indices(6) = 2; A.column_indices(6) = 1; A.values(6) = 7;

    A.sort();

    EXPECT_EQ(A.row_indices(0), 0); EXPECT_EQ(A.column_indices(0), 3); EXPECT_EQ(A.values(0), 6);
    EXPECT_EQ(A.row_indices(1), 1); EXPECT_EQ(A.column_indices(1), 2); EXPECT_EQ(A.values(1), 5);
    EXPECT_EQ(A.row_indices(2), 1); EXPECT_EQ(A.column_indices(2), 3); EXPECT_EQ(A.values(2), 3);
    EXPECT_EQ(A.row_indices(3), 2); EXPECT_EQ(A.column_indices(3), 1); EXPECT_EQ(A.values(3), 7);
    EXPECT_EQ(A.row_indices(4), 2); EXPECT_EQ(A.column_indices(4), 2); EXPECT_EQ(A.values(4), 4);
    EXPECT_EQ(A.row_indices(5), 3); EXPECT_EQ(A.column_indices(5), 1); EXPECT_EQ(A.values(5), 1);
    EXPECT_EQ(A.row_indices(6), 4); EXPECT_EQ(A.column_indices(6), 2); EXPECT_EQ(A.values(6), 2);
    // clang-format on
  } else {
    EXPECT_THROW(A.sort(), Morpheus::NotImplementedException);
  }
}

TYPED_TEST(CooMatrixUnaryTest, IsSortedByRow) {
  using Matrix = typename TestFixture::device;

  Matrix A;
  EXPECT_THROW(A.is_sorted_by_row(), Morpheus::NotImplementedException);
}

TYPED_TEST(CooMatrixUnaryTest, IsSorted) {
  using Matrix = typename TestFixture::device;
  using space  = typename Matrix::execution_space;

  Matrix A(5, 5, 7);

  if (Morpheus::is_serial_execution_space<space>::value) {
    // clang-format off
    A.row_indices(0) = 3; A.column_indices(0) = 1; A.values(0) = 1;
    A.row_indices(1) = 4; A.column_indices(1) = 2; A.values(1) = 2;
    A.row_indices(2) = 1; A.column_indices(2) = 3; A.values(2) = 3;
    A.row_indices(3) = 2; A.column_indices(3) = 2; A.values(3) = 4;
    A.row_indices(4) = 1; A.column_indices(4) = 2; A.values(4) = 5;
    A.row_indices(5) = 0; A.column_indices(5) = 3; A.values(5) = 6;
    A.row_indices(6) = 2; A.column_indices(6) = 1; A.values(6) = 7;
    // clang-format on

    EXPECT_FALSE(A.is_sorted());
    A.sort();
  } else {
    EXPECT_THROW(A.is_sorted(), Morpheus::NotImplementedException);
  }
}

/**
 * @brief Test Suite using the Compatible Binary CooMatrix pairs
 *
 */
TYPED_TEST_CASE(CompatibleCooMatrixBinaryTest, CompatibleCooMatrixBinary);

TYPED_TEST(CompatibleCooMatrixBinaryTest, ConstructionFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;

  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  Matrix1 Aref(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Aref, nrows, ncols, nnnz);

  HostMatrix1 Ahref(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ahref, nrows, ncols, nnnz);

  // clang-format off
  Ahref.row_indices(0) = 0; Ahref.column_indices(0) = 0; Ahref.values(0) = 1.11;
  Ahref.row_indices(1) = 0; Ahref.column_indices(1) = 2; Ahref.values(1) = 2.22;
  Ahref.row_indices(2) = 1; Ahref.column_indices(2) = 2; Ahref.values(2) = 3.33;
  Ahref.row_indices(3) = 2; Ahref.column_indices(3) = 1; Ahref.values(3) = 4.44;
  // clang-format on

  // Send to device
  Morpheus::copy(Ahref, Aref);

  // Build Matrix from the compatible Matrix
  Matrix2 A(Aref);
  CHECK_COO_CONTAINERS(A, Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), Ahref.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ahref.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref.values(n));
  }

  // Default copy construction
  HostMatrix1 Bh(Ah);
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
  Matrix1 B(A);
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

TYPED_TEST(CompatibleCooMatrixBinaryTest, CopyAssignmentFromCooMatrix) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using index_type  = typename Matrix1::index_type;

  // Matrix to Build
  // [1.11 *    2.22]
  // [*    *    3.33]
  // [*    4.44 *   ]
  index_type nrows = 3, ncols = 3, nnnz = 4;
  Matrix1 Aref(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Aref, nrows, ncols, nnnz);

  HostMatrix1 Ahref(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ahref, nrows, ncols, nnnz);

  // clang-format off
  Ahref.row_indices(0) = 0; Ahref.column_indices(0) = 0; Ahref.values(0) = 1.11;
  Ahref.row_indices(1) = 0; Ahref.column_indices(1) = 2; Ahref.values(1) = 2.22;
  Ahref.row_indices(2) = 1; Ahref.column_indices(2) = 2; Ahref.values(2) = 3.33;
  Ahref.row_indices(3) = 2; Ahref.column_indices(3) = 1; Ahref.values(3) = 4.44;
  // clang-format on

  // Send to device
  Morpheus::copy(Ahref, Aref);

  // Build Matrix from the compatible Matrix
  Matrix2 A = Aref;
  CHECK_COO_CONTAINERS(A, Aref);

  HostMatrix2 Ah(nrows, ncols, nnnz);
  CHECK_COO_SIZES(Ah, nrows, ncols, nnnz)
  Morpheus::copy(A, Ah);

  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Ah.row_indices(n), Ahref.row_indices(n));
    EXPECT_EQ(Ah.column_indices(n), Ahref.column_indices(n));
    EXPECT_EQ(Ah.values(n), Ahref.values(n));
  }

  // Default copy asssignment
  HostMatrix1 Bh = Ah;
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
  Matrix1 B = A;
  CHECK_COO_CONTAINERS(B, A);
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix1 Bt(nrows, ncols, nnnz);
  Morpheus::copy(B, Bt);
  for (index_type n = 0; n < nnnz; n++) {
    EXPECT_EQ(Bt.row_indices(n), Ah.row_indices(n));
    EXPECT_EQ(Bt.column_indices(n), Ah.column_indices(n));
    EXPECT_EQ(Bt.values(n), Ah.values(n));
  }
}

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