/**
 * Test_Multiply_Dynamic.hpp
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

#ifndef TEST_CORE_TEST_MULTIPLY_DYNAMIC_HPP
#define TEST_CORE_TEST_MULTIPLY_DYNAMIC_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_DenseVector.hpp>
#include <utils/Macros_DynamicMatrix.hpp>
#include <utils/MatrixGenerator.hpp>

using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;
using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using DynamicMatrixPairs =
    generate_pair<generate_pair<DynamicMatrixTypes, DenseVectorTypes>::type,
                  DenseVectorTypes>::type;

using DynamicMultiplyTypes = to_gtest_types<DynamicMatrixPairs>::type;

template <typename Containers>
class DynamicMultiplyTypesTest : public ::testing::Test {
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
  using SizeType          = typename mat_dev_t::size_type;
  using ValueType         = typename mat_dev_t::value_type;
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

      // Assign Coo Host matrix to Dynamic Host container
      mat_host_t Ah = Acoo;

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

TYPED_TEST_SUITE(DynamicMultiplyTypesTest, DynamicMultiplyTypes);

TYPED_TEST(DynamicMultiplyTypesTest, DynamicMultiplyCustomInit) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Create a duplicate of A on host
    auto Ah = Morpheus::create_mirror(c.A);
    Morpheus::copy(c.A, Ah);

    for (auto fmt_idx = 0; fmt_idx < Morpheus::NFORMATS; fmt_idx++) {
      // Convert to the new active state
      Morpheus::convert<Morpheus::Serial>(Ah, fmt_idx);
      auto A = Morpheus::create_mirror_container<backend>(Ah);
      Morpheus::copy(Ah, A);

      // Set to 1 to check if it will be initialized to 0
      vec2_t y(c.y.size(), 1);
      Morpheus::multiply<TEST_CUSTOM_SPACE>(A, c.x, y);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
      EXPECT_TRUE(Morpheus::Test::have_approx_same_data(y, c.y));
    }
  }
}

TYPED_TEST(DynamicMultiplyTypesTest, DynamicMultiplyGenericInit) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Create a duplicate of A on host
    auto Ah = Morpheus::create_mirror(c.A);
    Morpheus::copy(c.A, Ah);

    for (auto fmt_idx = 0; fmt_idx < Morpheus::NFORMATS; fmt_idx++) {
      // Convert to the new active state
      Morpheus::convert<Morpheus::Serial>(Ah, fmt_idx);
      auto A = Morpheus::create_mirror_container<backend>(Ah);
      Morpheus::copy(Ah, A);

      // Set to 1 to check if it will be initialized to 0
      vec2_t y(c.y.size(), 1);
      Morpheus::multiply<TEST_GENERIC_SPACE>(A, c.x, y);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
      EXPECT_TRUE(Morpheus::Test::have_approx_same_data(y, c.y));
    }
  }
}

TYPED_TEST(DynamicMultiplyTypesTest, DynamicMultiplyCustom) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Create a duplicate of A on host
    auto Ah = Morpheus::create_mirror(c.A);
    Morpheus::copy(c.A, Ah);

    for (auto fmt_idx = 0; fmt_idx < Morpheus::NFORMATS; fmt_idx++) {
      // Convert to the new active state
      Morpheus::convert<Morpheus::Serial>(Ah, fmt_idx);
      auto A = Morpheus::create_mirror_container<backend>(Ah);
      Morpheus::copy(Ah, A);

      vec2_t y(c.y.size(), 1);
      Morpheus::multiply<TEST_CUSTOM_SPACE>(A, c.x, y, false);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
      EXPECT_FALSE(Morpheus::Test::have_approx_same_data(y, c.y));

      auto yh = Morpheus::create_mirror_container(y);
      Morpheus::copy(y, yh);
      auto cyh = Morpheus::create_mirror(c.y);
      Morpheus::copy(c.y, cyh);
      for (size_type n = 0; n < cyh.size(); n++) {
        cyh(n) += 1;
      }
      EXPECT_TRUE(Morpheus::Test::have_approx_same_data(yh, cyh));
    }
  }
}

TYPED_TEST(DynamicMultiplyTypesTest, DynamicMultiplyGeneric) {
  using vec2_t    = typename TestFixture::vec2_dev_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;

  for (size_type i = 0; i < this->samples; i++) {
    auto c = this->containers[i];

    // Create a duplicate of A on host
    auto Ah = Morpheus::create_mirror(c.A);
    Morpheus::copy(c.A, Ah);

    for (auto fmt_idx = 0; fmt_idx < Morpheus::NFORMATS; fmt_idx++) {
      // Convert to the new active state
      Morpheus::convert<Morpheus::Serial>(Ah, fmt_idx);
      auto A = Morpheus::create_mirror_container<backend>(Ah);
      Morpheus::copy(Ah, A);

      vec2_t y(c.y.size(), 1);
      Morpheus::multiply<TEST_GENERIC_SPACE>(A, c.x, y, false);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(y));
      EXPECT_FALSE(Morpheus::Test::have_approx_same_data(y, c.y));

      auto yh = Morpheus::create_mirror_container(y);
      Morpheus::copy(y, yh);
      auto cyh = Morpheus::create_mirror(c.y);
      Morpheus::copy(c.y, cyh);
      for (size_type n = 0; n < cyh.size(); n++) {
        cyh(n) += 1;
      }
      EXPECT_TRUE(Morpheus::Test::have_approx_same_data(yh, cyh));
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_MULTIPLY_DYNAMIC_HPP
