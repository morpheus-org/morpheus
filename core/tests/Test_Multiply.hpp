/**
 * Test_Multiply.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef TEST_CORE_TEST_MULTIPLY_HPP
#define TEST_CORE_TEST_MULTIPLY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>
#include <utils/Macros_DenseMatrix.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>
#include <utils/Macros_EllMatrix.hpp>
#include <utils/Macros_HybMatrix.hpp>
#include <utils/Macros_HdcMatrix.hpp>
#include <utils/MatrixGenerator.hpp>

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::types_set>::type;
using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types::types_set>::type;
using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::types_set>::type;
using EllMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::EllMatrix<double>,
                                               types::types_set>::type;
using HybMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HybMatrix<double>,
                                               types::types_set>::type;
using HdcMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HdcMatrix<double>,
                                               types::types_set>::type;
using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using CooMatrixPairs =
    generate_pair<generate_pair<CooMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;
using CsrMatrixPairs =
    generate_pair<generate_pair<CsrMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;
using DiaMatrixPairs =
    generate_pair<generate_pair<DiaMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;
using EllMatrixPairs =
    generate_pair<generate_pair<EllMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;
using HybMatrixPairs =
    generate_pair<generate_pair<HybMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;
using HdcMatrixPairs =
    generate_pair<generate_pair<HdcMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;

using pairs = typename Morpheus::concat<
    CooMatrixPairs,
    typename Morpheus::concat<
        CsrMatrixPairs,
        typename Morpheus::concat<
            DiaMatrixPairs,
            typename Morpheus::concat<
                EllMatrixPairs,
                typename Morpheus::concat<HybMatrixPairs, HdcMatrixPairs>::
                    type>::type>::type>::type>::type;

using MultiplyTypes = to_gtest_types<pairs>::type;

template <typename Containers>
class MultiplyTypesTest : public ::testing::Test {
 public:
  using type              = Containers;
  using mat_container_t   = typename Containers::first_type::first_type::type;
  using vec1_container_t  = typename Containers::first_type::second_type::type;
  using vec2_container_t  = typename Containers::second_type::type;
  using mat_dev_t         = typename mat_container_t::type;
  using mat_host_t        = typename mat_container_t::type::HostMirror;
  using vec1_dev_t        = typename vec1_container_t::type;
  using vec1_host_t       = typename vec1_container_t::type::HostMirror;
  using vec2_dev_t        = typename vec2_container_t::type;
  using vec2_host_t       = typename vec2_container_t::type::HostMirror;
  using ValueType         = typename mat_dev_t::value_type;
  using SizeType          = typename mat_dev_t::size_type;
  using IndexType         = typename mat_dev_t::index_type;
  using ArrayLayout       = typename mat_dev_t::array_layout;
  using Backend           = typename mat_dev_t::backend;
  using MirrorArrayLayout = typename mat_host_t::array_layout;
  using MirrorBackend     = typename mat_host_t::backend;

  class ContainersClass {
   public:
    using diag_generator = Morpheus::Test::DiagonalMatrixGenerator<
        ValueType, IndexType, MirrorArrayLayout, MirrorBackend>;

    mat_dev_t A;
    vec1_dev_t x;
    vec2_dev_t y;
    ContainersClass() : A(), x(), y() {}

    ContainersClass(SizeType nrows, SizeType ncols,
                    std::vector<int>& diag_indexes)
        : A(), x(ncols, 0), y(nrows, 0) {
      // Generate the diagonal matrix
      diag_generator generator(nrows, ncols, diag_indexes);
      typename diag_generator::DenseMatrix Adense;
      typename diag_generator::SparseMatrix Acoo;
      Adense = generator.generate();
      generator.generate(Acoo);

      // Convert it in the format we are interested in
      mat_host_t Ah;
      Morpheus::convert<Morpheus::Serial>(Acoo, Ah);

      // Copy on device
      A.resize(Ah);
      Morpheus::copy(Ah, A);

      // Randomly assign the x vector
      unsigned long long seed = 5374857;
      Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);
      x.assign(x.size(), rand_pool, -1.0, 1.0);

      auto xh = Morpheus::create_mirror_container(x);
      Morpheus::copy(x, xh);

      auto yh = Morpheus::create_mirror_container(y);
      yh.assign(yh.size(), 0);

      for (SizeType i = 0; i < nrows; i++) {
        for (SizeType j = 0; j < ncols; j++) {
          yh(i) += Adense(i, j) * xh(j);
        }
      }
      Morpheus::copy(yh, y);
    }
  };

  static const SizeType samples = 3;
  SizeType nrows[samples]       = {10, 100, 1000};
  SizeType ncols[samples]       = {10, 100, 1000};

  ContainersClass containers[samples];

  void SetUp() override {
    for (SizeType i = 0; i < samples; i++) {
      int diag_freq = 5;
      int ndiags    = ((nrows[i] - 1) / diag_freq) * 2 + 1;
      std::vector<int> diags(ndiags, 0);

      diags[0]            = 0;
      SizeType diag_count = 1;
      for (int nd = 1; nd < (int)nrows[i]; nd++) {
        if (nd % diag_freq == 0) {
          diags[diag_count]     = nd;
          diags[diag_count + 1] = -nd;
          diag_count += 2;
        }
      }

      ContainersClass c(nrows[i], ncols[i], diags);
      containers[i] = c;
    }
  }
};

namespace Test {

TYPED_TEST_SUITE(MultiplyTypesTest, MultiplyTypes);

TYPED_TEST(MultiplyTypesTest, MultiplyCustomInit) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Set to 1 to check if it will be initialized to 0
    vec2_t y(c.y.size(), 1);
    Morpheus::multiply<TEST_CUSTOM_SPACE>(c.A, c.x, y);

    EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(y, c.y));
  }
}

TYPED_TEST(MultiplyTypesTest, MultiplyGenericInit) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Set to 1 to check if it will be initialized to 0
    vec2_t y(c.y.size(), 1);
    Morpheus::multiply<TEST_GENERIC_SPACE>(c.A, c.x, y);

    EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(y, c.y));
  }
}

TYPED_TEST(MultiplyTypesTest, MultiplyCustom) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    vec2_t y(c.y.size(), 1);
    Morpheus::multiply<TEST_CUSTOM_SPACE>(c.A, c.x, y, false);

    EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
    EXPECT_FALSE(Morpheus::Test::have_approx_same_data(y, c.y));

    auto yh = Morpheus::create_mirror_container(y);
    Morpheus::copy(y, yh);
    auto cyh = Morpheus::create_mirror_container(c.y);
    Morpheus::copy(c.y, cyh);
    for (size_type n = 0; n < cyh.size(); n++) {
      cyh(n) += 1;
    }
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(yh, cyh));
  }
}

TYPED_TEST(MultiplyTypesTest, MultiplyGeneric) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    vec2_t y(c.y.size(), 1);
    Morpheus::multiply<TEST_GENERIC_SPACE>(c.A, c.x, y, false);

    EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
    EXPECT_FALSE(Morpheus::Test::have_approx_same_data(y, c.y));

    auto yh = Morpheus::create_mirror_container(y);
    Morpheus::copy(y, yh);
    auto cyh = Morpheus::create_mirror_container(c.y);
    Morpheus::copy(c.y, cyh);
    for (size_type n = 0; n < cyh.size(); n++) {
      cyh(n) += 1;
    }
    EXPECT_TRUE(Morpheus::Test::have_approx_same_data(yh, cyh, true));
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_MULTIPLY_HPP
