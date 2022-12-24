/**
 * Test_Convert_Dynamic.hpp
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

#ifndef TEST_CORE_HIP_TEST_CONVERT_DYNAMIC_HPP
#define TEST_CORE_HIP_TEST_CONVERT_DYNAMIC_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

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
using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;

using CooMatrixPairs = generate_pair<DynamicMatrixTypes, CooMatrixTypes>::type;
using CsrMatrixPairs = generate_pair<DynamicMatrixTypes, CsrMatrixTypes>::type;
using DiaMatrixPairs = generate_pair<DynamicMatrixTypes, DiaMatrixTypes>::type;
using EllMatrixPairs = generate_pair<DynamicMatrixTypes, EllMatrixTypes>::type;
using HybMatrixPairs = generate_pair<DynamicMatrixTypes, HybMatrixTypes>::type;

using pairs = typename Morpheus::concat<
    CooMatrixPairs,
    typename Morpheus::concat<
        CsrMatrixPairs,
        typename Morpheus::concat<
            DiaMatrixPairs,
            typename Morpheus::concat<EllMatrixPairs, HybMatrixPairs>::type>::
            type>::type>::type;

using ConvertDynamicTypes = to_gtest_types<pairs>::type;

template <typename Containers>
class ConvertDynamicTypesTest : public ::testing::Test {
 public:
  using type       = Containers;
  using dyn_t      = typename Containers::first_type::type;
  using con_t      = typename Containers::second_type::type;
  using dyn_dev_t  = typename dyn_t::type;
  using dyn_host_t = typename dyn_t::type::HostMirror;
  using con_dev_t  = typename con_t::type;
  using con_host_t = typename con_t::type::HostMirror;
  using IndexType  = typename dyn_dev_t::index_type;
  using ValueType  = typename dyn_dev_t::value_type;
  using LayoutType = typename dyn_dev_t::array_layout;
  using SpaceType  = typename dyn_dev_t::backend;
  using coo_dev_t =
      Morpheus::CooMatrix<ValueType, IndexType, LayoutType, SpaceType>;
  using coo_host_t = typename coo_dev_t::HostMirror;
  using csr_dev_t =
      Morpheus::CsrMatrix<ValueType, IndexType, LayoutType, SpaceType>;
  using csr_host_t = typename csr_dev_t::HostMirror;

  void SetUp() override {
    Morpheus::Test::setup_small_container(con_ref_h);
    con_ref.resize(con_ref_h);
    Morpheus::copy(con_ref_h, con_ref);

    Morpheus::Test::setup_small_container(csr_ref_h);
    csr_ref.resize(csr_ref_h);
    Morpheus::copy(csr_ref_h, csr_ref);

    dyn_ref_h = csr_ref_h;
    dyn_ref   = csr_ref;
  }

  dyn_host_t dyn_ref_h;
  dyn_dev_t dyn_ref;

  csr_host_t csr_ref_h;
  csr_dev_t csr_ref;

  con_host_t con_ref_h;
  con_dev_t con_ref;
};

namespace Test {

TYPED_TEST_SUITE(ConvertDynamicTypesTest, ConvertDynamicTypes);

TYPED_TEST(ConvertDynamicTypesTest, DynamicToSparseHIP) {
  using src_t = typename TestFixture::dyn_dev_t;
  using dst_t = typename TestFixture::con_dev_t;
  using csr_t = typename TestFixture::csr_dev_t;

  dst_t dst;
  dst.resize(this->con_ref);

  csr_t csr_src;
  csr_src.resize(this->csr_ref);

  src_t src = csr_src;

  EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src, dst),
               Morpheus::NotImplementedException);
}

TYPED_TEST(ConvertDynamicTypesTest, SparseToDynamicHIP) {
  using src_t = typename TestFixture::con_dev_t;
  using dst_t = typename TestFixture::dyn_dev_t;

  src_t src = this->con_ref;

  dst_t dst;
  dst.resize(this->dyn_ref);

  for (int fmt_id = 0; fmt_id < Morpheus::NFORMATS; fmt_id++) {
    EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src, dst),
                 Morpheus::NotImplementedException);
  }
}

TYPED_TEST(ConvertDynamicTypesTest, DynamicToDynamicHIP) {
  using src_t = typename TestFixture::dyn_dev_t;
  using dst_t = typename TestFixture::dyn_dev_t;
  src_t src   = this->con_ref;
  dst_t dst;
  EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src, dst),
               Morpheus::NotImplementedException);
}

TYPED_TEST(ConvertDynamicTypesTest, DynamicInPlaceHIP) {
  using dst_t = typename TestFixture::dyn_dev_t;
  dst_t dst   = this->con_ref;
  for (int fmt_id = 0; fmt_id < Morpheus::NFORMATS; fmt_id++) {
    int state = dst.active_index();
    Morpheus::conversion_error_e status =
        Morpheus::convert<TEST_CUSTOM_SPACE>(dst, fmt_id);

    if (state == fmt_id) {
      EXPECT_EQ(status, Morpheus::CONV_SUCCESS);
    } else if (state == Morpheus::COO_FORMAT) {
      EXPECT_EQ(status, Morpheus::PROXY_TO_DYNAMIC);
    } else {
      EXPECT_EQ(status, Morpheus::DYNAMIC_TO_PROXY);
    }
  }
}
}  // namespace Test

#endif  // TEST_CORE_HIP_TEST_CONVERT_DYNAMIC_HPP
