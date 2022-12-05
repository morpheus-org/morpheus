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

#ifndef TEST_CORE_TEST_CONVERT_DYNAMIC_HPP
#define TEST_CORE_TEST_CONVERT_DYNAMIC_HPP

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

using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               types::types_set>::type;

using CooMatrixPairs = generate_pair<DynamicMatrixTypes, CooMatrixTypes>::type;
using CsrMatrixPairs = generate_pair<DynamicMatrixTypes, CsrMatrixTypes>::type;
using DiaMatrixPairs = generate_pair<DynamicMatrixTypes, DiaMatrixTypes>::type;

using pairs = typename Morpheus::concat<
    CooMatrixPairs,
    typename Morpheus::concat<CsrMatrixPairs, DiaMatrixPairs>::type>::type;

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

TYPED_TEST(ConvertDynamicTypesTest, DynamicToSparse) {
  using src_t      = typename TestFixture::dyn_dev_t;
  using src_host_t = typename TestFixture::dyn_host_t;
  using dst_t      = typename TestFixture::con_dev_t;
  using csr_t      = typename TestFixture::csr_dev_t;

  dst_t dst;
  dst.resize(this->con_ref);

  csr_t csr_src;
  csr_src.resize(this->csr_ref);

  src_t src = csr_src;

#if defined(MORPHEUS_ENABLE_SERIAL)
  if (Morpheus::has_serial_execution_space<TEST_CUSTOM_SPACE>::value) {
    auto dst_h       = Morpheus::create_mirror_container(dst);
    auto csr_src_h   = Morpheus::create_mirror_container(csr_src);
    src_host_t src_h = csr_src_h;
    Morpheus::copy(this->dyn_ref_h, src_h);

    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    Morpheus::Test::have_same_data(dst_h, this->con_ref_h);
  }
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  if (Morpheus::has_openmp_execution_space<TEST_CUSTOM_SPACE>::value) {
    auto dst_h       = Morpheus::create_mirror_container(dst);
    auto csr_src_h   = Morpheus::create_mirror_container(csr_src);
    src_host_t src_h = csr_src_h;
    Morpheus::copy(this->dyn_ref_h, src_h);

    if (Morpheus::has_same_format_v<csr_t, dst_t>) {
      Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
      Morpheus::Test::have_same_data(dst_h, this->con_ref_h);
    } else if (Morpheus::is_csr_matrix_format_container_v<csr_t> &&
               Morpheus::is_coo_matrix_format_container_v<dst_t>) {
      Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
      Morpheus::Test::have_same_data(dst_h, this->con_ref_h);
    } else {
      EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h),
                   Morpheus::NotImplementedException);
    }
  }
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  if (Morpheus::has_cuda_execution_space<TEST_CUSTOM_SPACE>::value) {
    EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src, dst),
                 Morpheus::NotImplementedException);
  }
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  if (Morpheus::has_hip_execution_space<TEST_CUSTOM_SPACE>::value) {
    EXPECT_THROW(Morpheus::convert<TEST_CUSTOM_SPACE>(src, dst),
                 Morpheus::NotImplementedException);
  }
#endif
}

TYPED_TEST(ConvertDynamicTypesTest, SparseToDynamic) {
  using src_t      = typename TestFixture::con_dev_t;
  using src_host_t = typename TestFixture::con_host_t;
  using dst_t      = typename TestFixture::dyn_dev_t;

  dst_t dst;
  dst.resize(this->dyn_ref);

  src_t src;
  src.resize(this->con_ref);

#if defined(MORPHEUS_ENABLE_SERIAL)
  if (Morpheus::has_serial_execution_space<TEST_CUSTOM_SPACE>::value) {
    auto dst_h = Morpheus::create_mirror_container(dst);
    auto src_h = Morpheus::create_mirror_container(src);
    Morpheus::copy(this->con_ref_h, src_h);

    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    // Check dynamic matrix has now same active type as original src
    EXPECT_EQ(src_h.format_index(), dst_h.active_index());

    // Convert back to concrete format and check with original src
    src_host_t ref_h;
    Morpheus::convert<TEST_CUSTOM_SPACE>(dst_h, ref_h);
    Morpheus::Test::have_same_data(src_h, ref_h);
  }
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
  if (Morpheus::has_openmp_execution_space<TEST_CUSTOM_SPACE>::value) {
    auto dst_h = Morpheus::create_mirror_container(dst);
    auto src_h = Morpheus::create_mirror_container(src);
    Morpheus::copy(this->con_ref_h, src_h);

    /* NOTE: Dst should always have the same format as src so we do not expect
     * to throw any NotImplemented errors
     */
    Morpheus::convert<TEST_CUSTOM_SPACE>(src_h, dst_h);
    // Check dynamic matrix has now same active type as original src
    EXPECT_EQ(src_h.format_index(), dst_h.active_index());

    // Convert back to concrete format and check with original src
    src_host_t ref_h;
    Morpheus::convert<TEST_CUSTOM_SPACE>(dst_h, ref_h);
    Morpheus::Test::have_same_data(src_h, ref_h);
  }
#endif  // MORPHEUS_ENABLE_OPENMP
}

// TODO
TYPED_TEST(ConvertDynamicTypesTest, DynamicToDynamic) { EXPECT_EQ(1, 1); }

// TODO: In-place conversion using format index.
}  // namespace Test

#endif  // TEST_CORE_TEST_CONVERT_HPP
