/**
 * Test_MirrorContainers.hpp
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
TEST(TESTSUITE_NAME, Mirror_DenseVector_HostMirror) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename vector::HostMirror>::value,
      "Mirror type should match the HostMirror type of the original container "
      "as we are creating a mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(x.size(), x_mirror.size());
    for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror should be the default "
                                   "(0) i.e no copy was performed";
    }
  }
}

TEST(TESTSUITE_NAME, Mirror_DenseVector_explicit_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<typename TEST_EXECSPACE::memory_space>,
      Kokkos::Cuda, Kokkos::DefaultHostExecutionSpace>;
  using dst_type =
      Morpheus::DenseVector<typename vector::value_type,
                            typename vector::index_type,
                            typename vector::array_layout, mirror_space>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror<mirror_space>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");

  ASSERT_EQ(x.size(), x_mirror.size());
  if (Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror should be the default "
                                   "(0) i.e no copy was performed";
    }
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_HostMirror) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror_container(x);
  using mirror  = decltype(x_mirror);

  static_assert(std::is_same<typename mirror::type,
                             typename vector::HostMirror::type>::value,
                "Source and mirror types must be the same as we are creating a "
                "mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename vector::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(x.size(), x_mirror.size());
    // Change the value to main container to check if we did shallow copy
    for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
      x[i] = -4;
    }

    for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -4) << "Value of the mirror should be equal to "
                                    "the new value of the vector "
                                    "container for shallow copy to be valid";
    }
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_same_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror_container<TEST_EXECSPACE>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename vector::type>::value,
      "Source and mirror types must be the same as we are creating a "
      "mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename vector::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(x.size(), x_mirror.size());
    // Change the value to main container to check if we did shallow copy
    for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
      x[i] = -4;
    }

    for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -4) << "Value of the mirror should be equal to "
                                    "the new value of the vector "
                                    "container for shallow copy to be valid";
    }
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_new_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<typename TEST_EXECSPACE::memory_space>,
      Kokkos::Cuda, Kokkos::DefaultHostExecutionSpace>;
  using dst_type =
      Morpheus::DenseVector<typename vector::value_type,
                            typename vector::index_type,
                            typename vector::array_layout, mirror_space>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror_container<mirror_space>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");
}

}  // namespace Test