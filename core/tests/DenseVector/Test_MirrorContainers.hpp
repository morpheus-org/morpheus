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

// Creates a mirror
// Checks shapes
// Checks that only allocation happened and values are not copied
// (Mirror always lives on host here)
TEST(TESTSUITE_NAME, Mirror_DenseVector_HostMirror) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using index_type = typename container::index_type;

  container x(10, -2);

  auto x_mirror = Morpheus::create_mirror(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type,
                   typename container::HostMirror>::value,
      "Mirror type should match the HostMirror type of the original container "
      "as we are creating a mirror in the same space.");

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});

  for (index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], 0)
        << "Value of the mirror values should be the default "
           "(0) i.e no copy was performed";
  }
}

// Creates a mirror container
// Checks shapes
// If both container and mirror are on host:
//  Shallow copy
// Otherwise only allocation
TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_HostMirror) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using index_type = typename container::index_type;

  container x(10, -2);

  auto x_mirror = Morpheus::create_mirror_container(x);
  using mirror  = decltype(x_mirror);

  static_assert(std::is_same<typename mirror::type,
                             typename container::HostMirror::type>::value,
                "Source and mirror types must be the same as we are creating a "
                "mirror in the same space.");

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});
  // Change the value to main container to check if we did shallow copy
  x.assign(x.size(), -4);

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -4)
          << "Value of the mirror should be equal to the new value of the "
             "vector container for shallow copy to be valid";
    }
  } else {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], 0)
          << "Value of the mirror values should be the default "
             "(0) i.e no copy was performed";
    }
  }
}

// Creates a mirror container in same space as container
// Checks shapes
// Checks that Shallow copy was performed for the mirror
// (only check on host as otherwise will return access error)
TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_same_space) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using index_type = typename container::index_type;

  container x(10, -2);

  auto x_mirror = Morpheus::create_mirror_container<TEST_EXECSPACE>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename container::type>::value,
      "Source and mirror types must be the same as we are creating a "
      "mirror in the same space.");

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});
  // Change the value to main container to check if we did shallow copy
  x.assign(x.size(), -4);

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -4)
          << "Value of the mirror should be equal to the new value of the "
             "vector container (-4) as a shallow copy was performed.";
    }
  }
}

#if defined(MORPHEUS_ENABLE_CUDA)

// Creates a mirror in explicit space
// Checks shapes
// If on host checks that only allocation happened and values are not copied
TEST(TESTSUITE_NAME, Mirror_DenseVector_explicit_space) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<test_memory_space>,
      typename Kokkos::Cuda::memory_space,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;
  using dst_type =
      Morpheus::DenseVector<typename container::value_type,
                            typename container::index_type,
                            typename container::array_layout, mirror_space>;
  using index_type = typename container::index_type;

  container x(10, -2);

  auto x_mirror = Morpheus::create_mirror<mirror_space>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});
  if (Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror values should be the "
                                   "default (0) i.e no copy was performed";
    }
  }
}

// Creates a mirror container in other space from container
// Checks types are the same for both mirror and container
TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_new_space) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<test_memory_space>,
      typename Kokkos::Cuda::memory_space,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;
  using dst_type =
      Morpheus::DenseVector<typename container::value_type,
                            typename container::index_type,
                            typename container::array_layout, mirror_space>;
  using index_type = typename container::index_type;

  container x(10, -2);

  auto x_mirror = Morpheus::create_mirror_container<mirror_space>(x);
  using mirror  = decltype(x_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");
}

#endif  // MORPHEUS_ENABLE_CUDA

}  // namespace Test