/**
 * Test_DenseMatrix.hpp
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

#ifndef TEST_CORE_TEST_DENSEMATRIX_HPP
#define TEST_CORE_TEST_DENSEMATRIX_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types::types_set>::type;
using DenseMatrixUnary = to_gtest_types<DenseMatrixTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DenseMatrixUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;
};

namespace Test {
/**
 * @brief Test Suite using the Unary DenseMatrix
 *
 */
TYPED_TEST_CASE(DenseMatrixUnaryTest, DenseMatrixUnary);

/**
 * @brief Testing default construction of DenseMatrix container
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, DefaultConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;

  Matrix A;
  EXPECT_EQ(A.nrows(), 0);
  EXPECT_EQ(A.ncols(), 0);
  EXPECT_EQ(A.nnnz(), 0);
  EXPECT_EQ(A.data(), nullptr);
  EXPECT_EQ(A.view().size(), 0);
  EXPECT_EQ(A.view().data(), nullptr);

  HostMatrix Ah;
  EXPECT_EQ(Ah.nrows(), 0);
  EXPECT_EQ(Ah.ncols(), 0);
  EXPECT_EQ(Ah.nnnz(), 0);
  EXPECT_EQ(Ah.data(), nullptr);
  EXPECT_EQ(Ah.view().size(), 0);
  EXPECT_EQ(Ah.view().data(), nullptr);
}

/**
 * @brief Testing default copy assignment of DenseMatrix container from another
 * DenseMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, DefaultCopyAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  auto nrows = 10, ncols = 15;
  Matrix A(nrows, ncols, (value_type)5.22);
  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);

  HostMatrix Ah(nrows, ncols, 0);
  Morpheus::copy(A, Ah);

  HostMatrix Bh = Ah;
  EXPECT_EQ(Ah.nrows(), Bh.nrows());
  EXPECT_EQ(Ah.ncols(), Bh.ncols());
  EXPECT_EQ(Ah.nnnz(), Bh.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device Matrix
  Matrix B = A;
  EXPECT_EQ(A.nrows(), B.nrows());
  EXPECT_EQ(A.ncols(), B.ncols());
  EXPECT_EQ(A.nnnz(), B.nnnz());
  Morpheus::copy(Ah, A);

  // Send other Matrix back to host for check
  HostMatrix Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

/**
 * @brief Testing default copy construction of DenseMatrix container from
 * another DenseMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, DefaultCopyConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  auto nrows = 10, ncols = 15;
  Matrix A(nrows, ncols, (value_type)5.22);
  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);

  HostMatrix Ah(nrows, ncols, 0);
  Morpheus::copy(A, Ah);

  HostMatrix Bh(Ah);
  EXPECT_EQ(Ah.nrows(), Bh.nrows());
  EXPECT_EQ(Ah.ncols(), Bh.ncols());
  EXPECT_EQ(Ah.nnnz(), Bh.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device Matrix
  Matrix B(A);
  EXPECT_EQ(A.nrows(), B.nrows());
  EXPECT_EQ(A.ncols(), B.ncols());
  EXPECT_EQ(A.nnnz(), B.nnnz());
  Morpheus::copy(Ah, A);

  // Send other Matrix back to host for check
  HostMatrix Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

/**
 * @brief Testing default move assignment of DenseMatrix container from another
 * DenseMatrix container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, DefaultMoveAssignment) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  auto nrows = 10, ncols = 15;
  Matrix A(nrows, ncols, (value_type)5.22);
  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);

  HostMatrix Ah(nrows, ncols, 0);
  Morpheus::copy(A, Ah);

  HostMatrix Bh = std::move(Ah);
  EXPECT_EQ(Ah.nrows(), Bh.nrows());
  EXPECT_EQ(Ah.ncols(), Bh.ncols());
  EXPECT_EQ(Ah.nnnz(), Bh.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device Matrix
  Matrix B = std::move(A);
  EXPECT_EQ(A.nrows(), B.nrows());
  EXPECT_EQ(A.ncols(), B.ncols());
  EXPECT_EQ(A.nnnz(), B.nnnz());
  Morpheus::copy(Ah, A);

  // Send other Matrix back to host for check
  HostMatrix Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

/**
 * @brief Testing default move construction of DenseMatrix container from
 * another DenseMatrix container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, DefaultMoveConstructor) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  auto nrows = 10, ncols = 15;
  Matrix A(nrows, ncols, (value_type)5.22);
  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);

  HostMatrix Ah(nrows, ncols, 0);
  Morpheus::copy(A, Ah);

  HostMatrix Bh(std::move(Ah));
  EXPECT_EQ(Ah.nrows(), Bh.nrows());
  EXPECT_EQ(Ah.ncols(), Bh.ncols());
  EXPECT_EQ(Ah.nnnz(), Bh.nnnz());

  Ah(4, 5) = (value_type)-4.33;
  Ah(9, 5) = (value_type)-9.44;

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), Bh(i, j));
    }
  }

  // Now check device Matrix
  Matrix B(std::move(A));
  EXPECT_EQ(A.nrows(), B.nrows());
  EXPECT_EQ(A.ncols(), B.ncols());
  EXPECT_EQ(A.nnnz(), B.nnnz());
  Morpheus::copy(Ah, A);

  // Send other Matrix back to host for check
  HostMatrix Bt(B.nrows(), B.ncols(), 0);
  Morpheus::copy(B, Bt);
  for (index_type i = 0; i < Bh.nrows(); i++) {
    for (index_type j = 0; j < Bh.ncols(); j++) {
      EXPECT_EQ(Bt(i, j), Bh(i, j));
    }
  }
}

/**
 * @brief Testing construction of DenseMatrix container with shape `num_rows *
 * num_cols` and values set at 0 by default
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, NormalConstructionDefaultVal) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  index_type num_rows = 10, num_cols = 15;
  value_type val = 0;

  Matrix A(num_rows, num_cols);
  HostMatrix Ah(num_rows, num_cols);

  EXPECT_EQ(A.nrows(), num_rows);
  EXPECT_EQ(A.ncols(), num_cols);
  EXPECT_EQ(A.nnnz(), num_rows * num_cols);
  EXPECT_EQ(Ah.nrows(), num_rows);
  EXPECT_EQ(Ah.ncols(), num_cols);
  EXPECT_EQ(Ah.nnnz(), num_rows * num_cols);

  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah.view()(i, j), val);
    }
  }
}

/**
 * @brief Testing construction of DenseMatrix container with shape `num_rows *
 * num_cols` and values set to `val`
 *
 */
TYPED_TEST(DenseMatrixUnaryTest, NormalConstruction) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  index_type num_rows = 10, num_cols = 15;
  value_type val = 15.22;

  Matrix A(num_rows, num_cols, val);
  HostMatrix Ah(num_rows, num_cols);

  EXPECT_EQ(A.nrows(), num_rows);
  EXPECT_EQ(A.ncols(), num_cols);
  EXPECT_EQ(A.nnnz(), num_rows * num_cols);
  EXPECT_EQ(Ah.nrows(), num_rows);
  EXPECT_EQ(Ah.ncols(), num_cols);
  EXPECT_EQ(Ah.nnnz(), num_rows * num_cols);

  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah.view()(i, j), val);
    }
  }
}

// /**
//  * @brief Testing construction of DenseMatrix from a raw pointer
//  *
//  */
// TYPED_TEST(DenseMatrixUnaryTest, PointerConstruction) { EXPECT_EQ(1, 0); }

TYPED_TEST(DenseMatrixUnaryTest, Assign) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;

  index_type nrows = 1000, ncols = 1500;

  Matrix A(nrows, ncols, 0);
  HostMatrix Ah(nrows, ncols, 0);

  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);
  EXPECT_EQ(Ah.nrows(), nrows);
  EXPECT_EQ(Ah.ncols(), ncols);
  EXPECT_EQ(Ah.nnnz(), nrows * ncols);

  A.assign(100, 150, (value_type)20.33);
  EXPECT_EQ(A.nrows(), 100);
  EXPECT_EQ(A.ncols(), 150);
  EXPECT_EQ(A.nnnz(), 100 * 150);
  Ah.assign(100, 150, (value_type)20.33);
  EXPECT_EQ(Ah.nrows(), 100);
  EXPECT_EQ(Ah.ncols(), 150);
  EXPECT_EQ(Ah.nnnz(), 100 * 150);
  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < A.nrows(); i++) {
    for (index_type j = 0; j < A.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), (value_type)20.33);
    }
  }

  A.assign(70, 80, (value_type)-30.11);
  EXPECT_EQ(A.nrows(), 70);
  EXPECT_EQ(A.ncols(), 80);
  EXPECT_EQ(A.nnnz(), 70 * 80);
  Ah.assign(70, 80, (value_type)-30.11);
  EXPECT_EQ(Ah.nrows(), 70);
  EXPECT_EQ(Ah.ncols(), 80);
  EXPECT_EQ(Ah.nnnz(), 70 * 80);
  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < A.nrows(); i++) {
    for (index_type j = 0; j < A.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), (value_type)-30.11);
    }
  }

  A.assign(nrows + 2000, ncols + 1500, (value_type)10.111);
  EXPECT_EQ(A.nrows(), nrows + 2000);
  EXPECT_EQ(A.ncols(), ncols + 1500);
  EXPECT_EQ(A.nnnz(), (nrows + 2000) * (ncols + 1500));

  Ah.assign(nrows + 2000, ncols + 1500, 0);
  EXPECT_EQ(Ah.nrows(), A.nrows());
  EXPECT_EQ(Ah.ncols(), A.ncols());
  EXPECT_EQ(Ah.nnnz(), A.nnnz());

  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), (value_type)10.111);
    }
  }
}

TYPED_TEST(DenseMatrixUnaryTest, Resize) {
  using Matrix     = typename TestFixture::device;
  using HostMatrix = typename TestFixture::host;
  using index_type = typename Matrix::index_type;

  index_type nrows = 1000, ncols = 1500;

  Matrix A(nrows, ncols, 0);
  HostMatrix Ah(nrows, ncols, 0);

  EXPECT_EQ(A.nrows(), nrows);
  EXPECT_EQ(A.ncols(), ncols);
  EXPECT_EQ(A.nnnz(), nrows * ncols);
  EXPECT_EQ(Ah.nrows(), nrows);
  EXPECT_EQ(Ah.ncols(), ncols);
  EXPECT_EQ(Ah.nnnz(), nrows * ncols);

  A.resize(100, 150);
  EXPECT_EQ(A.nrows(), 100);
  EXPECT_EQ(A.ncols(), 150);
  EXPECT_EQ(A.nnnz(), 100 * 150);
  Ah.resize(100, 150);
  EXPECT_EQ(Ah.nrows(), 100);
  EXPECT_EQ(Ah.ncols(), 150);
  EXPECT_EQ(Ah.nnnz(), 100 * 150);
  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < A.nrows(); i++) {
    for (index_type j = 0; j < A.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), 0);
    }
  }

  A.resize(70, 80);
  EXPECT_EQ(A.nrows(), 70);
  EXPECT_EQ(A.ncols(), 80);
  EXPECT_EQ(A.nnnz(), 70 * 80);
  Ah.resize(70, 80);
  EXPECT_EQ(Ah.nrows(), 70);
  EXPECT_EQ(Ah.ncols(), 80);
  EXPECT_EQ(Ah.nnnz(), 70 * 80);
  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < A.nrows(); i++) {
    for (index_type j = 0; j < A.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), 0);
    }
  }

  A.resize(nrows + 2000, ncols + 1500);
  EXPECT_EQ(A.nrows(), nrows + 2000);
  EXPECT_EQ(A.ncols(), ncols + 1500);
  EXPECT_EQ(A.nnnz(), (nrows + 2000) * (ncols + 1500));

  Ah.resize(nrows + 2000, ncols + 1500);
  EXPECT_EQ(Ah.nrows(), A.nrows());
  EXPECT_EQ(Ah.ncols(), A.ncols());
  EXPECT_EQ(Ah.nnnz(), A.nnnz());

  Morpheus::copy(A, Ah);

  for (index_type i = 0; i < Ah.nrows(); i++) {
    for (index_type j = 0; j < Ah.ncols(); j++) {
      EXPECT_EQ(Ah(i, j), 0);
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEMATRIX_HPP
