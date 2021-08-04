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

// // Create a mirror container in a new space (specialization for same space)
// template <class Space, template <class, class...> class Container, class T,
//           class... P>
// typename Impl::MirrorContainerType<Space, Container, T, P...>::container_type
// create_mirror_container(
//     const Container<T, P...>& src,
//     typename std::enable_if<Impl::MirrorContainerType<
//         Space, Container, T, P...>::is_same_memspace>::type* = nullptr);

// // Create a mirror DenseVector in a new space (specialization for different
// // space)
// template <class Space, class T, class... P>
// typename Impl::MirrorContainerType<Space, DenseVector, T,
// P...>::container_type create_mirror_container(
//     const DenseVector<T, P...>& src,
//     typename std::enable_if<!Impl::MirrorContainerType<
//         Space, DenseVector, T, P...>::is_same_memspace>::type* = nullptr);

namespace Test {
TEST(TESTSUITE_NAME, Mirror_DenseVector_same_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror(x);

  using mirror = decltype(x_mirror);
  ::testing::StaticAssertTypeEq<typename mirror::type,
                                typename vector::HostMirror>();
  ASSERT_EQ(x.size(), x_mirror.size());

  for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror should be the default "
                                 "(0) i.e no copy was performed";
  }
}

TEST(TESTSUITE_NAME, Mirror_DenseVector_explicit_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;
  using mirror_space = Kokkos::DefaultHostExecutionSpace;
  using res_vector =
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft, mirror_space>;

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror<mirror_space>(x);
  using mirror  = decltype(x_mirror);

  ::testing::StaticAssertTypeEq<
      typename Morpheus::Impl::MirrorType<mirror_space, Morpheus::DenseVector,
                                          float, long long, Kokkos::LayoutLeft,
                                          TEST_EXECSPACE>::container_type,
      typename mirror::type>();

  ASSERT_EQ(x.size(), x_mirror.size());

  for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], 0) << "Value of the mirror should be the default "
                                 "(0) i.e no copy was performed";
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_same_space) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       Kokkos::DefaultHostExecutionSpace>;
  using mirror_space = Kokkos::DefaultHostExecutionSpace;
  using res_vector =
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft, mirror_space>;

  static_assert(
      std::is_same<typename vector::memory_space,
                   typename vector::HostMirror::memory_space>::value &&
          std::is_same<typename vector::value_type,
                       typename vector::HostMirror::value_type>::value,
      "Source and mirror space must be the same and vectors should have the "
      "same value type");

  vector x(10, -2);
  auto x_mirror = Morpheus::create_mirror_container(x);
  using mirror  = decltype(x_mirror);

  ::testing::StaticAssertTypeEq<typename mirror::type,
                                typename vector::HostMirror::type>();

  ASSERT_EQ(x.size(), x_mirror.size());

  // Change the value to main container to check if we did shallow copy
  for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
    x[i] = -4;
  }

  for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], -4)
        << "Value of the mirror should be equal to the new value of the vector "
           "container for shallow copy to be valid";
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_same_space) {}

TEST(TESTSUITE_NAME, MirrorContainer_DenseVector_explicit_new_space) {}

}  // namespace Test