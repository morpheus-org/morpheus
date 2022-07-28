/**
 * Test_DenseMatrix_CompatibleBinary.hpp
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

#ifndef TEST_CORE_TEST_DENSEMATRIX_COMPATIBLEBINARY_HPP
#define TEST_CORE_TEST_DENSEMATRIX_COMPATIBLEBINARY_HPP

#include <Morpheus_Core.hpp>
#include <Utils.hpp>

using DenseMatrixCompatibleTypes = typename Morpheus::generate_unary_typelist<
    Morpheus::DenseMatrix<double>, types::compatible_types_set>::type;

using Compatible DenseMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DenseMatrixCompatibleTypes, DenseMatrixCompatibleTypes>::type>::type;

// Used for testing compatible binary operations
template <typename BinaryContainer>
class CompatibleDenseMatrixBinaryTest : public ::testing::Test {
 public:
  using type  = BinaryContainer;
  using type1 = typename BinaryContainer::type1;  // Unary
  using type2 = typename BinaryContainer::type2;  // Unary

  using device1 = typename type1::type;  // DenseMatrix
  using host1   = typename type1::type::HostMirror;

  using device2 = typename type2::type;  // DenseMatrix
  using host2   = typename type2::type::HostMirror;
};

namespace Test {
/**
 * @brief Test Suite using the Compatible Binary DenseMatrix
 *
 */
TYPED_TEST_CASE(CompatibleDenseMatrixBinaryTest, CompatibleDenseMatrixBinary);

/**
 * @brief Testing shallow copy construction of DenseMatrix container from
 * another DenseMatrix container with the different parameters. Resulting
 * container should be a shallow copy of the original.
 *
 */
TYPED_TEST(CompatibleDenseMatrixBinaryTest, ShallowCopyConstructor) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix1::value_type;
  using index_type  = typename Matrix1::index_type;

  index_type N0 = 10, N1 = 15;
  Matrix1 A(N0, N1, (value_type)5.22);
  EXPECT_EQ(A.nrows(), N0);
  EXPECT_EQ(A.ncols(), N1);
  EXPECT_EQ(A.nnnz(), N0 * N1);

  HostMatrix1 Ah(A.nrows(), A.cols(), (value_type)0);
  Morpheus::copy(A, Ah);

  HostMatrix2 Bh(Ah);
  EXPECT_EQ(Bh.nrows(), Ah.nrows());
  EXPECT_EQ(Bh.ncols(), Ah.ncols());
  EXPECT_EQ(Bh.nnnz(), Ah.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device vector
  Matrix2 B(A);
  EXPECT_EQ(B.nrows(), A.nrows());
  EXPECT_EQ(B.ncols(), A.ncols());
  EXPECT_EQ(B.nnnz(), A.nnnz());
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix2 Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

/**
 * @brief Testing shallow copy assignment of DenseMatrix container from
 * another DenseMatrix container with the different parameters. Resulting
 * copy should be a shallow copy from the original.
 *
 */
TYPED_TEST(CompatibleDenseMatrixBinaryTest, ShallowCopyAssignment) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix1::value_type;
  using index_type  = typename Matrix1::index_type;

  index_type N0 = 10, N1 = 15;
  Matrix1 A(N0, N1, (value_type)5.22);
  EXPECT_EQ(A.nrows(), N0);
  EXPECT_EQ(A.ncols(), N1);
  EXPECT_EQ(A.nnnz(), N0 * N1);

  HostMatrix1 Ah(A.nrows(), A.cols(), (value_type)0);
  Morpheus::copy(A, Ah);

  HostMatrix2 Bh = Ah;
  EXPECT_EQ(Bh.nrows(), Ah.nrows());
  EXPECT_EQ(Bh.ncols(), Ah.ncols());
  EXPECT_EQ(Bh.nnnz(), Ah.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device vector
  Matrix2 B = A;
  EXPECT_EQ(B.nrows(), A.nrows());
  EXPECT_EQ(B.ncols(), A.ncols());
  EXPECT_EQ(B.nnnz(), A.nnnz());
  Morpheus::copy(Ah, A);

  // Send other vector back to host for check
  HostMatrix2 Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEMATRIX_COMPATIBLEBINARY_HPP
