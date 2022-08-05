/**
 * Test_Convert.hpp
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

#ifndef TEST_CORE_TEST_CONVERT_HPP
#define TEST_CORE_TEST_CONVERT_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types::types_set>::type;

using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::types_set>::type;

using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types::types_set>::type;

using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::types_set>::type;

using DenseMatrixCooMatrixPairs =
    generate_pair<DenseMatrixTypes, CooMatrixTypes>::type;

using DenseMatrixCsrMatrixPairs =
    generate_pair<DenseMatrixTypes, CsrMatrixTypes>::type;

using DenseMatrixDiaMatrixPairs =
    generate_pair<DenseMatrixTypes, DiaMatrixTypes>::type;

using pairs = typename Morpheus::concat<
    DenseMatrixCooMatrixPairs,
    typename Morpheus::concat<DenseMatrixCsrMatrixPairs,
                              DenseMatrixDiaMatrixPairs>::type>::type;
using ConvertTypes = to_gtest_types<pairs>::type;

template <typename Containers>
class ConvertTest : public ::testing::Test {
 public:
  using type          = Containers;
  using source_t      = typename Containers::first_type::type;
  using dest_t        = typename Containers::second_type::type;
  using source_device = typename source_t::type;
  using source_host   = typename source_t::type::HostMirror;
  using dest_device   = typename dest_t::type;
  using dest_host     = typename dest_t::type::HostMirror;
  using IndexType     = typename source_device::index_type;
  using ValueType     = typename source_device::value_type;

  void SetUp() override {
    source_host Aref_h(1000, 1000, 0.0);
    nnnz = 0;
    for (IndexType i = 0; i < Aref_h.nrows(); i++) {
      for (IndexType j = 0; j < Aref_h.ncols(); j++) {
        if ((i == j) || (std::abs(j - i) == 2) ||
            (i > 5 && i < 15 && j > 5 && j < 15)) {
          Aref_h(i, j) = ValueType(i * Aref_h.ncols() + j);
          nnnz++;
        }
      }
    }

    source_device Aref(Aref_h.nrows(), Aref_h.ncols());
    Morpheus::copy(Aref_h, Aref);

    ref   = Aref;
    ref_h = Aref_h;
  }

  IndexType nnnz;
  source_host ref_h;
  source_device ref;
};

namespace Test {

TYPED_TEST_CASE(ConvertTest, ConvertTypes);

TYPED_TEST(ConvertTest, ForwardThenBackward) {
  using src_t      = typename TestFixture::source_host;
  using dst_t      = typename TestFixture::dest_host;
  using index_type = typename src_t::index_type;
  using value_type = typename src_t::value_type;

  dst_t A;
  src_t Adense;
  bool validate = false;
#if defined(MORPHEUS_ENABLE_CUDA)
  if (Morpheus::is_cuda_execution_space<TEST_EXECSPACE>::value) {
    EXPECT_THROW(Morpheus::convert<TEST_EXECSPACE>(this->ref, A),
                 Morpheus::NotImplementedException);
  }
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  if (Morpheus::is_openmp_execution_space<TEST_EXECSPACE>::value) {
    EXPECT_THROW(Morpheus::convert<TEST_EXECSPACE>(this->ref, A),
                 Morpheus::NotImplementedException);
  }
#endif

  if (Morpheus::is_serial_execution_space<TEST_EXECSPACE>::value) {
    Morpheus::convert<TEST_EXECSPACE>(this->ref, A);
    Morpheus::convert<TEST_EXECSPACE>(A, Adense);
    validate = true;
  }

  if (validate) {
    typename src_t::HostMirror Adense_h(Adense.nrows(), Adense.ncols());
    Morpheus::copy(Adense, Adense_h);

    for (index_type i = 0; i < this->ref_h.nrows(); i++) {
      for (index_type j = 0; j < this->ref_h.ncols(); j++) {
        if (std::is_floating_point<value_type>::value) {
          EXPECT_PRED_FORMAT2(
              ::testing::internal::CmpHelperFloatingPointEQ<value_type>,
              this->ref_h(i, j), Adense_h(i, j));
        } else {
          EXPECT_EQ(this->ref_h(i, j), Adense_h(i, j));
        }
      }
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_CONVERT_HPP
