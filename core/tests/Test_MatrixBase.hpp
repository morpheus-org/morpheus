/**
 * Test_MatrixBase.hpp
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

#ifndef TEST_CORE_TEST_MATRIXBASE_HPP
#define TEST_CORE_TEST_MATRIXBASE_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

template <class ValueType, class... Properties>
class TestContainer
    : public Morpheus::MatrixBase<TestContainer, ValueType, Properties...> {
 public:
  using traits =
      Morpheus::ContainerTraits<TestContainer, ValueType, Properties...>;
  using type = typename traits::type;
  using base = Morpheus::MatrixBase<TestContainer, ValueType, Properties...>;
  using index_type = typename traits::index_type;

  TestContainer(index_type n, index_type m, index_type nnz) : base(n, m, nnz) {}
};

using MatrixBaseTypes =
    typename Morpheus::generate_unary_typelist<TestContainer<double>,
                                               types::types_set>::type;
using MatrixBaseUnary = to_gtest_types<MatrixBaseTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class MatrixBaseTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using matrix = typename UnaryContainer::type;
  using base   = typename matrix::base;
};

namespace Test {

/**
 * @brief Test Suite using the MatrixBase
 *
 */
TYPED_TEST_SUITE(MatrixBaseTest, MatrixBaseUnary);

/**
 * @brief Checks if the container has specific traits attached to it.
 *
 */
TYPED_TEST(MatrixBaseTest, CheckTraits) {
  using Base = typename TestFixture::base;

  EXPECT_EQ(Morpheus::has_type<Base>::value, 1);
  EXPECT_EQ(Morpheus::has_traits<Base>::value, 1);
  EXPECT_EQ(Morpheus::has_index_type<Base>::value, 1);
}

/**
 * @brief Constructs a new container using the default constructor
 *
 */
TYPED_TEST(MatrixBaseTest, DefaultConstruct) {
  using Base = typename TestFixture::base;

  Base M;

  EXPECT_EQ(M.nrows(), 0);
  EXPECT_EQ(M.ncols(), 0);
  EXPECT_EQ(M.nnnz(), 0);
  EXPECT_EQ(M.structure(), Morpheus::MATSTR_NONE);
  EXPECT_EQ(M.options(), Morpheus::MATSTR_NONE);
}

/**
 * @brief Constructs a new container from a provided shape
 *
 */
TYPED_TEST(MatrixBaseTest, ConstructFromShape) {
  using Base       = typename TestFixture::base;
  using index_type = typename Base::index_type;

  index_type nrows = 3, ncols = 5, nnnz = 4;
  {
    Morpheus::MatrixStructure str = Morpheus::MATSTR_NONE;
    Morpheus::MatrixOptions opt   = Morpheus::MATOPT_NONE;

    Base M(nrows, ncols, nnnz);

    EXPECT_EQ(M.nrows(), nrows);
    EXPECT_EQ(M.ncols(), ncols);
    EXPECT_EQ(M.nnnz(), nnnz);
    EXPECT_EQ(M.structure(), opt);
    EXPECT_EQ(M.options(), str);
  }

  {
    Morpheus::MatrixStructure str = Morpheus::MATSTR_SYMMETRIC;
    Morpheus::MatrixOptions opt   = Morpheus::MATOPT_SHORT_ROWS;

    Base M(nrows, ncols, nnnz);

    M.set_structure(Morpheus::MATSTR_SYMMETRIC);
    M.set_options(Morpheus::MATOPT_SHORT_ROWS);

    EXPECT_EQ(M.nrows(), nrows);
    EXPECT_EQ(M.ncols(), ncols);
    EXPECT_EQ(M.nnnz(), nnnz);
    EXPECT_EQ(M.structure(), str);
    EXPECT_EQ(M.options(), opt);
  }
}

/**
 * @brief Resizes the container from a provided shape
 *
 */
TYPED_TEST(MatrixBaseTest, ResizeFromShape) {
  using Base       = typename TestFixture::base;
  using index_type = typename Base::index_type;

  index_type nrows = 3, ncols = 5, nnnz = 4;

  Base M;

  M.resize(nrows, ncols, nnnz);

  EXPECT_EQ(M.nrows(), nrows);
  EXPECT_EQ(M.ncols(), ncols);
  EXPECT_EQ(M.nnnz(), nnnz);
}

/**
 * @brief Ensures setter member functions work as expected
 *
 */
TYPED_TEST(MatrixBaseTest, CheckSetters) {
  using Base       = typename TestFixture::base;
  using index_type = typename Base::index_type;

  index_type nrows = 3, ncols = 5, nnnz = 4;
  Morpheus::MatrixStructure str = Morpheus::MATSTR_SYMMETRIC;
  Morpheus::MatrixOptions opt   = Morpheus::MATOPT_SHORT_ROWS;

  Base M;

  M.set_nrows(nrows);
  M.set_ncols(ncols);
  M.set_nnnz(nnnz);
  M.set_structure(str);
  M.set_options(opt);

  EXPECT_EQ(M.nrows(), nrows);
  EXPECT_EQ(M.ncols(), ncols);
  EXPECT_EQ(M.nnnz(), nnnz);
  EXPECT_EQ(M.structure(), str);
  EXPECT_EQ(M.options(), opt);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_MATRIXBASE_HPP
