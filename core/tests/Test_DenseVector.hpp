/**
 * Test_DenseVector.hpp
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

#include <Morpheus_Core.hpp>

namespace Test {
TEST(TESTSUITE_NAME, DenseVectorInstantiation_dlr) {
  using vector = Morpheus::DenseVector<double, long long, Kokkos::LayoutRight,
                                       TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<double, long long, Kokkos::LayoutRight,
                            TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, double>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type,
                                double>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutRight>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_dll) {
  using vector = Morpheus::DenseVector<double, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<double, long long, Kokkos::LayoutLeft,
                            TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, double>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type,
                                double>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_dir) {
  using vector =
      Morpheus::DenseVector<double, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<double, int, Kokkos::LayoutRight,
                            TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, double>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type,
                                double>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutRight>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_dil) {
  using vector =
      Morpheus::DenseVector<double, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<double, int, Kokkos::LayoutLeft, TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, double>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type,
                                double>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_flr) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutRight,
                                       TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, long long, Kokkos::LayoutRight,
                            TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutRight>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_fll) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                            TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_fir) {
  using vector =
      Morpheus::DenseVector<float, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, int, Kokkos::LayoutRight, TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutRight>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_fil) {
  using vector =
      Morpheus::DenseVector<float, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, int, Kokkos::LayoutLeft, TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_f___) {
  using vector = Morpheus::DenseVector<float>;

  ::testing::StaticAssertTypeEq<typename vector::type,
                                Morpheus::DenseVector<float>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<
      typename vector::array_layout,
      typename Kokkos::DefaultExecutionSpace::array_layout>();

  ::testing::StaticAssertTypeEq<
      typename vector::memory_space,
      typename Kokkos::DefaultExecutionSpace::memory_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::execution_space,
      typename Kokkos::DefaultExecutionSpace::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_f_l_) {
  using vector = Morpheus::DenseVector<float, Kokkos::LayoutLeft>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, Kokkos::LayoutLeft>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                typename Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<
      typename vector::memory_space,
      typename Kokkos::DefaultExecutionSpace::memory_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::execution_space,
      typename Kokkos::DefaultExecutionSpace::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_fl_s) {
  using vector = Morpheus::DenseVector<float, long long, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, long long, TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                typename TEST_EXECSPACE::array_layout>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_fil_) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, long long>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type,
                                long long>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                typename Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<
      typename vector::memory_space,
      typename Kokkos::DefaultExecutionSpace::memory_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::execution_space,
      typename Kokkos::DefaultExecutionSpace::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

TEST(TESTSUITE_NAME, DenseVectorInstantiation_f_ls) {
  using vector =
      Morpheus::DenseVector<float, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  ::testing::StaticAssertTypeEq<
      typename vector::type,
      Morpheus::DenseVector<float, Kokkos::LayoutLeft, TEST_EXECSPACE>>();
  ::testing::StaticAssertTypeEq<typename vector::tag,
                                Morpheus::DenseVectorTag>();

  ::testing::StaticAssertTypeEq<typename vector::value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_value_type, float>();
  ::testing::StaticAssertTypeEq<typename vector::size_type, size_t>();
  ::testing::StaticAssertTypeEq<typename vector::index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::non_const_index_type, int>();
  ::testing::StaticAssertTypeEq<typename vector::array_layout,
                                typename Kokkos::LayoutLeft>();

  ::testing::StaticAssertTypeEq<typename vector::memory_space,
                                typename TEST_EXECSPACE::memory_space>();
  ::testing::StaticAssertTypeEq<typename vector::execution_space,
                                typename TEST_EXECSPACE::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename vector::device_type,
      Kokkos::Device<typename vector::execution_space,
                     typename vector::memory_space>>();

  ::testing::StaticAssertTypeEq<
      typename vector::HostMirror,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         typename vector::host_mirror_space::memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename vector::host_mirror_type,
      Morpheus::DenseVector<
          typename vector::non_const_value_type,
          typename vector::non_const_index_type, typename vector::array_layout,
          typename vector::host_mirror_space::memory_space>>();

  ::testing::StaticAssertTypeEq<typename vector::pointer,
                                std::add_pointer<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_pointer,
      std::add_pointer<std::add_const<vector>::type>::type>();
  ::testing::StaticAssertTypeEq<typename vector::reference,
                                std::add_lvalue_reference<vector>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::const_reference,
      std::add_lvalue_reference<std::add_const<vector>::type>::type>();

  ::testing::StaticAssertTypeEq<typename vector::value_array_type,
                                Kokkos::View<typename vector::value_type*,
                                             typename vector::device_type>>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_pointer,
      std::add_pointer<typename vector::value_type>::type>();
  ::testing::StaticAssertTypeEq<
      typename vector::value_array_reference,
      std::add_lvalue_reference<typename vector::value_type>::type>();
}

}  // namespace Test