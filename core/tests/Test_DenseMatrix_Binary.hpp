/**
 * Test_DenseMatrix_Binary.hpp
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

#ifndef TEST_CORE_TEST_DENSEMATRIX_BINARY_HPP
#define TEST_CORE_TEST_DENSEMATRIX_BINARY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types::types_set>::type;

using DenseMatrixBinary =
    to_gtest_types<typename Morpheus::generate_binary_typelist<
        DenseMatrixTypes, DenseMatrixTypes>::type>::type;

// Used for testing binary operations
template <typename BinaryContainer>
class DenseMatrixBinaryTest : public ::testing::Test {
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
 * @brief Test Suite using the Binary DenseMatrix
 *
 */
TYPED_TEST_SUITE(DenseMatrixBinaryTest, DenseMatrixBinary);

/**
 * @brief Testing allocation of DenseMatrix container from another DenseMatrix
 * container with the different parameters. New allocation shouldn't alias the
 * original.
 *
 */
TYPED_TEST(DenseMatrixBinaryTest, Allocate) {
  using Matrix1     = typename TestFixture::device1;
  using HostMatrix1 = typename TestFixture::host1;
  using Matrix2     = typename TestFixture::device2;
  using HostMatrix2 = typename TestFixture::host2;
  using value_type  = typename Matrix1::value_type;
  using size_type   = typename Matrix1::size_type;

  size_type N0 = 10, N1 = 15;
  Matrix1 A(N0, N1, (value_type)5.22);
  EXPECT_EQ(A.nrows(), N0);
  EXPECT_EQ(A.ncols(), N1);
  EXPECT_EQ(A.nnnz(), N0 * N1);

  HostMatrix1 Ah(A.nrows(), A.ncols(), 0);
  Morpheus::copy(A, Ah);

  HostMatrix2 Bh;
  EXPECT_EQ(Bh.nrows(), 0);
  EXPECT_EQ(Bh.ncols(), 0);
  EXPECT_EQ(Bh.nnnz(), 0);
  EXPECT_EQ(Bh.view().size(), 0);

  Bh.allocate(Ah);
  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  EXPECT_EQ(Bh.nrows(), Ah.nrows());
  EXPECT_EQ(Bh.ncols(), Ah.ncols());
  EXPECT_EQ(Bh.nnnz(), Ah.nnnz());
  EXPECT_EQ(Bh.view().size(), Ah.view().size());
  for (size_type i = 0; i < Ah.nrows(); i++) {
    for (size_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Bh(i, j), (value_type)0);
    }
  }

  // Now check device Matrix
  Matrix2 B;
  EXPECT_EQ(B.nrows(), 0);
  EXPECT_EQ(B.ncols(), 0);
  EXPECT_EQ(B.nnnz(), 0);
  EXPECT_EQ(B.view().size(), 0);

  Bh(4, 5) = (value_type)-4.33;
  Bh(9, 5) = (value_type)-9.44;
  B.allocate(A);
  Morpheus::copy(B, Bh);

  EXPECT_EQ(B.nrows(), Bh.nrows());
  EXPECT_EQ(B.ncols(), Bh.ncols());
  EXPECT_EQ(B.nnnz(), Bh.nnnz());
  EXPECT_EQ(B.view().size(), Bh.view().size());
  for (size_type i = 0; i < Ah.nrows(); i++) {
    for (size_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Bh(i, j), (value_type)0);
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEMATRIX_BINARY_HPP
