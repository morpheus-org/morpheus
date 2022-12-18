/**
 * Test_MatrixOperations_Dynamic.hpp
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

#ifndef TEST_CORE_TEST_MATRIXOPERATIONS_DYNAMIC_HPP
#define TEST_CORE_TEST_MATRIXOPERATIONS_DYNAMIC_HPP

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
    generate_pair<DynamicMatrixTypes, DenseVectorTypes>::type;

using pairs = DynamicMatrixPairs;

using DynamicMatrixOperationsTypes = to_gtest_types<pairs>::type;

template <typename Containers>
class DynamicMatrixOperationsTypesTest : public ::testing::Test {
 public:
  using type              = Containers;
  using mat_container_t   = typename Containers::first_type::type;
  using vec_container_t   = typename Containers::second_type::type;
  using mat_dev_t         = typename mat_container_t::type;
  using mat_host_t        = typename mat_container_t::type::HostMirror;
  using vec_dev_t         = typename vec_container_t::type;
  using vec_host_t        = typename vec_container_t::type::HostMirror;
  using SizeType          = typename mat_dev_t::size_type;
  using ValueType         = typename mat_dev_t::value_type;
  using IndexType         = typename mat_dev_t::index_type;
  using ArrayLayout       = typename mat_dev_t::array_layout;
  using Backend           = typename mat_dev_t::backend;
  using MirrorArrayLayout = typename mat_host_t::array_layout;
  using MirrorBackend     = typename mat_host_t::backend;

  class ContainersClass {
   public:
    using diag_generator = Morpheus::Test::SparseDiagonalMatrixGenerator<
        ValueType, IndexType, MirrorArrayLayout, MirrorBackend>;

    mat_dev_t A;
    vec_host_t ref_diag;

    ContainersClass() : A(), ref_diag() {}

    ContainersClass(SizeType nrows, SizeType ncols,
                    std::vector<int>& diag_indexes)
        : A(), ref_diag(ncols, 0) {
      // Generate the diagonal matrix
      diag_generator generator(nrows, ncols, diag_indexes);
      typename diag_generator::DenseMatrix Adense;
      typename diag_generator::SparseMatrix Acoo;
      Adense = generator.generate();
      generator.generate(Acoo);

      for (SizeType i = 0; i < Adense.ncols(); i++) {
        ref_diag(i) = Adense(i, i);
      }

      // Assign Coo Host matrix to Dynamic Host container
      mat_host_t Ah = Acoo;

      // Copy on device
      A.resize(Ah);
      Morpheus::copy(Ah, A);
    }
  };

  static const int samples = 3;
  SizeType nrows[samples]  = {10, 100, 1000};
  SizeType ncols[samples]  = {10, 100, 1000};

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

TYPED_TEST_SUITE(DynamicMatrixOperationsTypesTest,
                 DynamicMatrixOperationsTypes);

TYPED_TEST(DynamicMatrixOperationsTypesTest, DynamicUpdateDiagonalCustom) {
  using vec_t     = typename TestFixture::vec_dev_t;
  using vec_h_t   = typename TestFixture::vec_host_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;
  using dense_mat_h_t =
      typename TestFixture::ContainersClass::diag_generator::DenseMatrix;
  using coo_mat_h_t =
      typename TestFixture::ContainersClass::diag_generator::SparseMatrix;

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

      vec_t new_diag(c.A.nrows(), 1);

      Morpheus::update_diagonal<TEST_CUSTOM_SPACE>(A, new_diag);

      coo_mat_h_t Acoo;
      dense_mat_h_t Adense;
      Morpheus::convert<Morpheus::Serial>(Ah, Acoo);
      Morpheus::convert<Morpheus::Serial>(Acoo, Adense);

      vec_h_t diag(Adense.nrows(), 0);

      for (size_type row = 0; row < Adense.nrows(); row++) {
        diag[row] = Adense(row, row);
      }

      vec_h_t new_diag_h = Morpheus::create_mirror_container(new_diag);
      Morpheus::copy(new_diag, new_diag_h);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(diag));
      for (size_type idx = 0; idx < diag.size(); idx++) {
        if (c.ref_diag[idx] == 0) {
          EXPECT_EQ(diag[idx], c.ref_diag[idx]);
        } else {
          EXPECT_EQ(diag[idx], new_diag_h[idx]);
        }
      }
    }
  }
}

TYPED_TEST(DynamicMatrixOperationsTypesTest, DynamicUpdateDiagonalGeneric) {
  using vec_t     = typename TestFixture::vec_dev_t;
  using vec_h_t   = typename TestFixture::vec_host_t;
  using size_type = typename TestFixture::SizeType;
  using backend   = typename TestFixture::Backend;
  using dense_mat_h_t =
      typename TestFixture::ContainersClass::diag_generator::DenseMatrix;
  using coo_mat_h_t =
      typename TestFixture::ContainersClass::diag_generator::SparseMatrix;

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

      vec_t new_diag(c.A.nrows(), 1);

      Morpheus::update_diagonal<TEST_GENERIC_SPACE>(A, new_diag);

      coo_mat_h_t Acoo;
      dense_mat_h_t Adense;
      Morpheus::convert<Morpheus::Serial>(Ah, Acoo);
      Morpheus::convert<Morpheus::Serial>(Acoo, Adense);

      vec_h_t diag(Adense.nrows(), 0);

      for (size_type row = 0; row < Adense.nrows(); row++) {
        diag[row] = Adense(row, row);
      }

      vec_h_t new_diag_h = Morpheus::create_mirror_container(new_diag);
      Morpheus::copy(new_diag, new_diag_h);

      EXPECT_FALSE(Morpheus::Test::is_empty_container(diag));
      for (size_type idx = 0; idx < diag.size(); idx++) {
        if (c.ref_diag[idx] == 0) {
          EXPECT_EQ(diag[idx], c.ref_diag[idx]);
        } else {
          EXPECT_EQ(diag[idx], new_diag_h[idx]);
        }
      }
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_MATRIXOPERATIONS_DYNAMIC_HPP
