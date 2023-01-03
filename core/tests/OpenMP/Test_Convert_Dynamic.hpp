/**
 * Test_Convert_Dynamic.hpp
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

#ifndef TEST_CORE_OPENMP_TEST_CONVERT_DYNAMIC_HPP
#define TEST_CORE_OPENMP_TEST_CONVERT_DYNAMIC_HPP

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
using HdcMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HdcMatrix<double>,
                                               types::types_set>::type;
using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;

using CooMatrixPairs = generate_pair<DynamicMatrixTypes, CooMatrixTypes>::type;
using CsrMatrixPairs = generate_pair<DynamicMatrixTypes, CsrMatrixTypes>::type;
using DiaMatrixPairs = generate_pair<DynamicMatrixTypes, DiaMatrixTypes>::type;
using EllMatrixPairs = generate_pair<DynamicMatrixTypes, EllMatrixTypes>::type;
using HybMatrixPairs = generate_pair<DynamicMatrixTypes, HybMatrixTypes>::type;
using HdcMatrixPairs = generate_pair<DynamicMatrixTypes, HdcMatrixTypes>::type;

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

TYPED_TEST(ConvertDynamicTypesTest, DynamicToSparseOpenMP) {
  using src_t      = typename TestFixture::dyn_dev_t;
  using src_host_t = typename TestFixture::dyn_host_t;
  using dst_t      = typename TestFixture::con_dev_t;
  using csr_t      = typename TestFixture::csr_dev_t;

  dst_t dst;
  dst.resize(this->con_ref);

  csr_t csr_src;
  csr_src.resize(this->csr_ref);

  src_t src = csr_src;

  auto dst_h       = Morpheus::create_mirror_container(dst);
  auto csr_src_h   = Morpheus::create_mirror_container(csr_src);
  src_host_t src_h = csr_src_h;
  Morpheus::copy(this->dyn_ref_h, src_h);

  if (Morpheus::has_same_format_v<csr_t, dst_t>) {
    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    EXPECT_TRUE(Morpheus::Test::have_same_data(dst_h, this->con_ref_h));
  } else if (Morpheus::is_csr_matrix_format_container_v<csr_t> &&
             Morpheus::is_coo_matrix_format_container_v<dst_t>) {
    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    EXPECT_TRUE(Morpheus::Test::have_same_data(dst_h, this->con_ref_h));
  } else {
    EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h),
                 Morpheus::NotImplementedException);
  }
}

TYPED_TEST(ConvertDynamicTypesTest, SparseToDynamicOpenMP) {
  using src_t      = typename TestFixture::con_dev_t;
  using src_host_t = typename TestFixture::con_host_t;
  using dst_t      = typename TestFixture::dyn_dev_t;

  dst_t dst;
  dst.resize(this->dyn_ref);

  src_t src;
  src.resize(this->con_ref);

  auto dst_h = Morpheus::create_mirror_container(dst);
  // Switch to COO such that conversions are from CSR to COO
  dst_h.activate(Morpheus::COO_FORMAT);

  auto src_h = Morpheus::create_mirror_container(src);
  Morpheus::copy(this->con_ref_h, src_h);

  if (Morpheus::is_dia_matrix_format_container_v<src_host_t> ||
      Morpheus::is_ell_matrix_format_container_v<src_host_t> ||
      Morpheus::is_hyb_matrix_format_container_v<src_host_t> ||
      Morpheus::is_hdc_matrix_format_container_v<src_host_t>) {
    EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h),
                 Morpheus::NotImplementedException);
  } else {
    auto fmt_idx = dst_h.active_index();

    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    // Check dynamic matrix has now same active type as original src
    EXPECT_EQ(fmt_idx, dst_h.active_index());

    // Convert back to concrete format and check with original src
    src_host_t ref_h;
    if (Morpheus::is_csr_matrix_format_container_v<src_host_t>) {
      // Convert on Serial backend since OMP not implemented yet
      Morpheus::convert<Morpheus::Serial>(dst_h, ref_h);
    } else {
      Morpheus::convert<TEST_CUSTOM_SPACE>(dst_h, ref_h);
    }
    EXPECT_TRUE(Morpheus::Test::have_same_data(src_h, ref_h));
  }
}

TYPED_TEST(ConvertDynamicTypesTest, DynamicToDynamicOpenMP) {
  using src_host_t = typename TestFixture::dyn_host_t;
  using dst_host_t = typename TestFixture::dyn_host_t;
  using coo_h_t    = typename TestFixture::coo_host_t;
  src_host_t src_h = this->con_ref_h;
  dst_host_t dst_h;

  for (int fmt_id = 0; fmt_id < Morpheus::NFORMATS; fmt_id++) {
    dst_h.activate(fmt_id);
    if (src_h.active_index() == dst_h.active_index() ||
        (src_h.active_index() == Morpheus::CSR_FORMAT &&
         dst_h.active_index() == Morpheus::COO_FORMAT)) {
      Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);

      EXPECT_TRUE(dst_h.active_index() == fmt_id);
      EXPECT_FALSE(Morpheus::Test::is_empty_container(dst_h));
      EXPECT_FALSE(Morpheus::Test::is_empty_container(src_h));
      // Convert back to COO to test correctness
      coo_h_t coo_src_h, coo_dst_h;

      Morpheus::convert<Morpheus::Serial>(src_h, coo_src_h);
      Morpheus::convert<Morpheus::Serial>(dst_h, coo_dst_h);
      EXPECT_TRUE(Morpheus::Test::have_same_data(coo_src_h, coo_dst_h));
    } else {
      EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h),
                   Morpheus::NotImplementedException);
    }
  }
}

TYPED_TEST(ConvertDynamicTypesTest, DynamicInPlaceOpenMP) {
  using dst_host_t = typename TestFixture::dyn_host_t;
  using coo_h_t    = typename TestFixture::coo_host_t;
  dst_host_t dst_h = this->con_ref_h;
  coo_h_t coo_ref_h;
  Morpheus::convert<Morpheus::Serial>(this->con_ref_h, coo_ref_h);

  for (int fmt_id = 0; fmt_id < Morpheus::NFORMATS; fmt_id++) {
    if (dst_h.active_index() == fmt_id ||
        (dst_h.active_index() == Morpheus::CSR_FORMAT &&
         fmt_id == Morpheus::COO_FORMAT)) {
      Morpheus::conversion_error_e status =
          Morpheus::convert<TEST_CUSTOM_SPACE>(dst_h, fmt_id);

      EXPECT_EQ(status, Morpheus::CONV_SUCCESS);
      EXPECT_TRUE(dst_h.active_index() == fmt_id);
      EXPECT_FALSE(Morpheus::Test::is_empty_container(dst_h));

      // Convert back to Concrete to test correctness
      coo_h_t coo_dst_h;
      Morpheus::convert<Morpheus::Serial>(dst_h, coo_dst_h);
      EXPECT_TRUE(Morpheus::Test::have_same_data(coo_ref_h, coo_dst_h));
    } else {
      int state = dst_h.active_index();
      Morpheus::conversion_error_e status =
          Morpheus::convert<TEST_CUSTOM_SPACE>(dst_h, fmt_id);

      if (((state == Morpheus::DIA_FORMAT) &&
           (fmt_id != Morpheus::DIA_FORMAT)) ||
          ((state == Morpheus::ELL_FORMAT) &&
           (fmt_id != Morpheus::ELL_FORMAT)) ||
          ((state == Morpheus::HYB_FORMAT) &&
           (fmt_id != Morpheus::HYB_FORMAT)) ||
          ((state == Morpheus::HDC_FORMAT) &&
           (fmt_id != Morpheus::HDC_FORMAT))) {
        EXPECT_EQ(status, Morpheus::DYNAMIC_TO_PROXY);
      } else {
        EXPECT_EQ(status, Morpheus::PROXY_TO_DYNAMIC);
      }
    }
  }
}
}  // namespace Test

#endif  // TEST_CORE_OPENMP_TEST_CONVERT_DYNAMIC_HPP
