/**
 * Test_Copy.hpp
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

#ifndef TEST_CORE_TEST_COPY_HPP
#define TEST_CORE_TEST_COPY_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_CsrMatrix.hpp>
#include <utils/Macros_DiaMatrix.hpp>
#include <utils/Macros_EllMatrix.hpp>
#include <utils/Macros_HybMatrix.hpp>
#include <utils/Macros_DenseMatrix.hpp>
#include <utils/Macros_DenseVector.hpp>
#include <utils/Macros_DynamicMatrix.hpp>

using value_tlist  = Morpheus::TypeList<double, int>;
using index_tlist  = Morpheus::TypeList<Morpheus::Default>;
using layout_tlist = Morpheus::TypeList<Morpheus::Default>;
using space_tlist  = Morpheus::TypeList<TEST_CUSTOM_SPACE>;
// Generate all unary combinations
using copy_types_set = typename Morpheus::cross_product<
    value_tlist,
    typename Morpheus::cross_product<
        index_tlist, typename Morpheus::cross_product<
                         layout_tlist, space_tlist>::type>::type>::type;

// Generate all unary combinations for every container and combine
using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               copy_types_set>::type;
using DenseMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseMatrix<double>,
                                               copy_types_set>::type;
using CooMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CooMatrix<double>,
                                               copy_types_set>::type;
using CsrMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::CsrMatrix<double>,
                                               copy_types_set>::type;
using DiaMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DiaMatrix<double>,
                                               copy_types_set>::type;
using EllMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::EllMatrix<double>,
                                               copy_types_set>::type;
using HybMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::HybMatrix<double>,
                                               copy_types_set>::type;
using DynamicMatrixTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DynamicMatrix<double>,
                                               copy_types_set>::type;

using CopyTypes = Morpheus::concat<
    DenseVectorTypes,
    Morpheus::concat<
        DenseMatrixTypes,
        Morpheus::concat<
            CooMatrixTypes,
            Morpheus::concat<
                CsrMatrixTypes,
                Morpheus::concat<
                    DiaMatrixTypes,
                    Morpheus::concat<
                        EllMatrixTypes,
                        Morpheus::concat<HybMatrixTypes, DynamicMatrixTypes>::
                            type>::type>::type>::type>::type>::type>::type;

using CopyTypesUnary       = to_gtest_types<CopyTypes>::type;
using CopyVectorTypesUnary = to_gtest_types<DenseVectorTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class CopyTypesUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;

  CopyTypesUnaryTest() : Aref(), Ahref() {}

  void SetUp() override {
    Morpheus::Test::setup_small_container(Ahref);
    Aref.resize(Ahref);
    Morpheus::copy(Ahref, Aref);
  }

  device Aref;
  host Ahref;
};

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class CopyVectorTypesUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;

  CopyVectorTypesUnaryTest() : Aref(), Ahref() {}

  void SetUp() override {
    Morpheus::Test::setup_small_container(Ahref);
    Aref.resize(Ahref);
    Morpheus::copy(Ahref, Aref);
  }

  device Aref;
  host Ahref;
};

namespace Test {

TYPED_TEST_SUITE(CopyTypesUnaryTest, CopyTypesUnary);

TYPED_TEST(CopyTypesUnaryTest, DeepCopyHost) {
  using HostContainer = typename TestFixture::host;

  HostContainer Ah;
  Ah.resize(this->Aref);

  Morpheus::copy(this->Ahref, Ah);

  // Update Ah
  Morpheus::Test::update_small_container(Ah);

  // Values of Ah should also be updated
  bool res = Morpheus::Test::have_same_data(Ah, this->Ahref);
  EXPECT_EQ(res, 0);
}

TYPED_TEST(CopyTypesUnaryTest, DeepCopyMirrorHost) {
  auto Ah = Morpheus::create_mirror_container(this->Ahref);

  Morpheus::copy(this->Ahref, Ah);

  bool res = Morpheus::Test::have_same_data(Ah, this->Ahref);
  EXPECT_EQ(res, 1);

  // Update Ah
  Morpheus::Test::update_small_container(Ah);

  // Values of Ah should also be updated
  res = Morpheus::Test::have_same_data(Ah, this->Ahref);
  EXPECT_EQ(res, 1);
}

TYPED_TEST(CopyTypesUnaryTest, DeepCopyDevice) {
  using DevContainer  = typename TestFixture::device;
  using HostContainer = typename TestFixture::host;

  DevContainer A;
  A.resize(this->Aref);

  HostContainer Bh;
  Bh.resize(this->Aref);

  auto Ah = Morpheus::create_mirror_container(A);

  // Copy Aref -> A
  Morpheus::copy(this->Aref, A);

  // Copy A -> Ah
  Morpheus::copy(A, Ah);

  // Copy A -> Bh
  Morpheus::copy(A, Bh);

  bool res = Morpheus::Test::have_same_data(Ah, Bh);
  EXPECT_EQ(res, 1);

  // Update Ah
  Morpheus::Test::update_small_container(Ah);

  if (Morpheus::has_host_execution_space_v<DevContainer>) {
    res = Morpheus::Test::have_same_data(Ah, A);
    EXPECT_EQ(res, 1);
  }
}

TYPED_TEST_SUITE(CopyVectorTypesUnaryTest, CopyVectorTypesUnary);
TYPED_TEST(CopyVectorTypesUnaryTest, CopyByKey) {
  using DevContainer  = typename TestFixture::device;
  using HostContainer = typename TestFixture::host;
  using size_type     = typename DevContainer::size_type;
  using value_type    = typename DevContainer::value_type;
  using index_type    = typename DevContainer::index_type;
  using array_layout  = typename DevContainer::array_layout;
  using space         = typename DevContainer::backend;
  using KeyVec =
      Morpheus::DenseVector<index_type, index_type, array_layout, space>;

  DevContainer src(20, 0), dst(20, 0);
  KeyVec keys(10, 0);

  // Set src & keys on host
  auto src_h = Morpheus::create_mirror_container(src);
  for (size_type i = 0; i < src.size(); i++) {
    src_h[i] = 1.11 * (value_type)i;
  }

  auto keys_h = Morpheus::create_mirror_container(keys);
  for (size_type kid = 0; kid < keys.size(); kid++) {
    keys_h[kid] = kid * 2;
  }

  // Copy src & keys on device
  Morpheus::copy(src_h, src);
  Morpheus::copy(keys_h, keys);

  // Copy by key
  Morpheus::copy_by_key<TEST_CUSTOM_SPACE>(keys, src, dst);

  // Check result
  HostContainer dst_ref_h(20, 0);
  for (size_type i = 0; i < keys.size(); i++) {
    dst_ref_h[i] = 1.11 * (value_type)(i * 2);
  }

  auto dst_h = Morpheus::create_mirror_container(dst);
  Morpheus::copy(dst, dst_h);

  bool res = Morpheus::Test::have_same_data(dst_h, dst_ref_h);
  EXPECT_EQ(res, 1);
}

TYPED_TEST(CopyVectorTypesUnaryTest, PartialCopySrc) {
  using DevContainer  = typename TestFixture::device;
  using HostContainer = typename TestFixture::host;
  using value_type    = typename DevContainer::value_type;
  using index_type    = typename DevContainer::index_type;

  index_type begin = 5, end = 13;
  DevContainer src(20, 0), dst(20, 0);

  // Set src on host
  auto src_h = Morpheus::create_mirror_container(src);
  for (auto i = 0; i < (int)src.size(); i++) {
    src_h[i] = 1.11 * (value_type)i;
  }

  // Copy src on device
  Morpheus::copy(src_h, src);

  // Copy range
  Morpheus::copy(src, dst, begin, end);

  // Check result
  HostContainer dst_ref_h(20, 0);
  for (auto i = begin; i < end; i++) {
    dst_ref_h[i] = 1.11 * (value_type)i;
  }

  auto dst_h = Morpheus::create_mirror_container(dst);
  Morpheus::copy(dst, dst_h);

  bool res = Morpheus::Test::have_same_data(dst_h, dst_ref_h);
  EXPECT_EQ(res, 1);
}

TYPED_TEST(CopyVectorTypesUnaryTest, PartialCopySrcDst) {
  using DevContainer  = typename TestFixture::device;
  using HostContainer = typename TestFixture::host;
  using value_type    = typename DevContainer::value_type;
  using index_type    = typename DevContainer::index_type;

  index_type src_begin = 5, src_end = 13, dst_begin = 2, dst_end = 10;
  DevContainer src(20, 0), dst(20, 0);

  // Set src on host
  auto src_h = Morpheus::create_mirror_container(src);
  for (auto i = 0; i < (int)src.size(); i++) {
    src_h[i] = 1.11 * (value_type)i;
  }

  // Copy src on device
  Morpheus::copy(src_h, src);

  // Copy range
  Morpheus::copy(src, dst, src_begin, src_end, dst_begin, dst_end);

  // Check result
  HostContainer dst_ref_h(20, 0);
  for (auto i = dst_begin; i < dst_end; i++) {
    dst_ref_h[i] = 1.11 * (value_type)(i + 3);
  }

  auto dst_h = Morpheus::create_mirror_container(dst);
  Morpheus::copy(dst, dst_h);

  bool res = Morpheus::Test::have_same_data(dst_h, dst_ref_h);
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_COPY_HPP