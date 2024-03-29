/**
 * Test_Convert.hpp
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

#ifndef TEST_CORE_SERIAL_TEST_CONVERT_HPP
#define TEST_CORE_SERIAL_TEST_CONVERT_HPP

#include <Morpheus_Core.hpp>

#include <utils/Utils.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>
#include <utils/Macros_EllMatrix.hpp>
#include <utils/Macros_HybMatrix.hpp>
#include <utils/Macros_HdcMatrix.hpp>
#include <utils/Macros_DenseMatrix.hpp>
#include <utils/Macros_DenseVector.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::convert_types_set>::type;
using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               types::convert_types_set>::type;
using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               types::convert_types_set>::type;
using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               types::convert_types_set>::type;
using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               types::convert_types_set>::type;
using EllMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::EllMatrix<double>,
                                               types::convert_types_set>::type;
using HybMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HybMatrix<double>,
                                               types::convert_types_set>::type;
using HdcMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HdcMatrix<double>,
                                               types::convert_types_set>::type;

using DenseMatrixCooMatrixPairs =
    generate_pair<DenseMatrixTypes, CooMatrixTypes>::type;
using CsrMatrixCooMatrixPairs =
    generate_pair<CsrMatrixTypes, CooMatrixTypes>::type;
using DiaMatrixCooMatrixPairs =
    generate_pair<DiaMatrixTypes, CooMatrixTypes>::type;
using EllMatrixCooMatrixPairs =
    generate_pair<EllMatrixTypes, CooMatrixTypes>::type;
using HybMatrixCooMatrixPairs =
    generate_pair<HybMatrixTypes, CooMatrixTypes>::type;
using HdcMatrixCooMatrixPairs =
    generate_pair<HdcMatrixTypes, CooMatrixTypes>::type;

using CooMatrixPairs = generate_pair<CooMatrixTypes, CooMatrixTypes>::type;
using CsrMatrixPairs = generate_pair<CsrMatrixTypes, CsrMatrixTypes>::type;
using DiaMatrixPairs = generate_pair<DiaMatrixTypes, DiaMatrixTypes>::type;
using EllMatrixPairs = generate_pair<EllMatrixTypes, EllMatrixTypes>::type;
using HybMatrixPairs = generate_pair<HybMatrixTypes, HybMatrixTypes>::type;
using HdcMatrixPairs = generate_pair<HdcMatrixTypes, HdcMatrixTypes>::type;
using DenseMatrixPairs =
    generate_pair<DenseMatrixTypes, DenseMatrixTypes>::type;
using DenseVectorPairs =
    generate_pair<DenseVectorTypes, DenseVectorTypes>::type;

using pairs = typename Morpheus::concat<
    DenseMatrixCooMatrixPairs,
    typename Morpheus::concat<
        CsrMatrixCooMatrixPairs,
        typename Morpheus::concat<
            DiaMatrixCooMatrixPairs,
            typename Morpheus::concat<
                EllMatrixCooMatrixPairs,
                typename Morpheus::concat<
                    HybMatrixCooMatrixPairs,
                    typename Morpheus::concat<
                        HdcMatrixCooMatrixPairs,
                        typename Morpheus::concat<
                            CooMatrixPairs,
                            typename Morpheus::concat<
                                CsrMatrixPairs,
                                typename Morpheus::concat<
                                    DiaMatrixPairs,
                                    typename Morpheus::concat<
                                        EllMatrixPairs,
                                        typename Morpheus::concat<
                                            HybMatrixPairs,
                                            typename Morpheus::concat<
                                                HdcMatrixPairs,
                                                typename Morpheus::concat<
                                                    DenseMatrixPairs,
                                                    DenseVectorPairs>::type>::
                                                type>::type>::type>::type>::
                                type>::type>::type>::type>::type>::type>::
        type>::type;

using ConvertTypes = to_gtest_types<pairs>::type;

template <typename Containers>
class ConvertTypesTest : public ::testing::Test {
 public:
  using type          = Containers;
  using source_t      = typename Containers::first_type::type;
  using dest_t        = typename Containers::second_type::type;
  using source_device = typename source_t::type;
  using source_host   = typename source_t::type::HostMirror;
  using dest_device   = typename dest_t::type;
  using dest_host     = typename dest_t::type::HostMirror;
  using ValueType     = typename source_device::value_type;

  void SetUp() override {
    Morpheus::Test::setup_small_container(src_ref_h);
    src_ref.resize(src_ref_h);
    Morpheus::copy(src_ref_h, src_ref);

    Morpheus::Test::setup_small_container(dst_ref_h);
    dst_ref.resize(dst_ref_h);
    Morpheus::copy(dst_ref_h, dst_ref);
  }

  source_host src_ref_h;
  source_device src_ref;

  dest_host dst_ref_h;
  dest_device dst_ref;
};

namespace Test {

TYPED_TEST_SUITE(ConvertTypesTest, ConvertTypes);

TYPED_TEST(ConvertTypesTest, ForwardSerial) {
  using src_t = typename TestFixture::source_device;
  using dst_t = typename TestFixture::dest_device;

  dst_t dst;
  dst.resize(this->dst_ref);

  src_t src;
  src.resize(this->src_ref);

  auto dst_h = Morpheus::create_mirror_container(dst);
  auto src_h = Morpheus::create_mirror_container(src);

  Morpheus::copy(this->src_ref_h, src_h);

  Morpheus::convert<Morpheus::Serial>(src_h, dst_h);
  Morpheus::Test::have_same_data(dst_h, this->dst_ref_h);
}

TYPED_TEST(ConvertTypesTest, BackwardSerial) {
  using src_t = typename TestFixture::dest_device;
  using dst_t = typename TestFixture::source_device;

  dst_t dst;
  dst.resize(this->src_ref);

  src_t src;
  src.resize(this->dst_ref);

  auto dst_h = Morpheus::create_mirror_container(dst);
  auto src_h = Morpheus::create_mirror_container(src);
  Morpheus::copy(this->dst_ref_h, src_h);

  Morpheus::convert<Morpheus::Serial>(src_h, dst_h);
  Morpheus::Test::have_same_data(dst_h, this->src_ref_h);
}

}  // namespace Test

#endif  // TEST_CORE_SERIAL_TEST_CONVERT_HPP
