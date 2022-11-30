/**
 * Test_ContainerTraits.hpp
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

#ifndef TEST_CORE_TEST_CONTAINERTRAITS_HPP
#define TEST_CORE_TEST_CONTAINERTRAITS_HPP

#include <Morpheus_Core.hpp>
#include <utils/Utils.hpp>

template <class ValueType, class... Properties>
class TestContainer : public Morpheus::ContainerTraits<TestContainer, ValueType,
                                                       Properties...> {
 public:
  using traits =
      Morpheus::ContainerTraits<TestContainer, ValueType, Properties...>;
};

using TestContainerTypes =
    typename Morpheus::generate_unary_typelist<TestContainer<double>,
                                               types::types_set>::type;

using TestContainerUnary = to_gtest_types<TestContainerTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class ContainerTraitsUnaryTest : public ::testing::Test {
 public:
  using type             = UnaryContainer;
  using container_traits = typename UnaryContainer::type::traits;
  using container_type   = typename UnaryContainer::type;
  //   using host   = typename UnaryContainer::type::HostMirror;
};

namespace Test {

TYPED_TEST_SUITE(ContainerTraitsUnaryTest, TestContainerUnary);

TYPED_TEST(ContainerTraitsUnaryTest, TestValueType) {
  using traits = typename TestFixture::container_traits;
  using unary  = typename TestFixture::type;

  EXPECT_TRUE((Morpheus::has_same_value_type_v<traits, unary>));
}

TYPED_TEST(ContainerTraitsUnaryTest, TestIndexType) {
  using traits      = typename TestFixture::container_traits;
  using unary       = typename TestFixture::type;
  using default_int = int;

  if (Morpheus::is_default_v<typename unary::index_type>) {
    EXPECT_TRUE(
        (std::is_same<typename traits::index_type, default_int>::value));
  } else {
    EXPECT_TRUE((Morpheus::has_same_index_type_v<traits, unary>));
  }
}

TYPED_TEST(ContainerTraitsUnaryTest, TestBackend) {
  using traits = typename TestFixture::container_traits;
  using unary  = typename TestFixture::type;
  using space  = typename unary::backend;

  using default_exe  = Kokkos::DefaultExecutionSpace;
  using default_mem  = typename default_exe::memory_space;
  using default_back = Morpheus::DefaultExecutionSpace;

  if (Morpheus::is_default_v<typename unary::backend>) {
    EXPECT_TRUE(
        (std::is_same_v<typename traits::execution_space, default_exe>));
    EXPECT_TRUE((std::is_same_v<typename traits::memory_space, default_mem>));
    EXPECT_TRUE((std::is_same_v<typename traits::backend, default_back>));
  } else {
    if (!Morpheus::has_backend_v<space>) {
      // Means we are using Kokkos::<spaces>
      EXPECT_TRUE((std::is_same_v<typename traits::backend,
                                  Morpheus::GenericBackend<space>>));
    } else {
      EXPECT_TRUE((std::is_same_v<typename traits::backend, space>));
    }
    using exe = typename traits::execution_space;
    using mem = typename traits::memory_space;

    EXPECT_TRUE((std::is_same_v<exe, typename space::execution_space>));
    EXPECT_TRUE((std::is_same_v<mem, typename space::memory_space>));
  }

  EXPECT_TRUE((std::is_same_v<typename traits::device_type,
                              Morpheus::Device<typename space::execution_space,
                                               typename space::memory_space,
                                               typename space::backend>>));

  EXPECT_TRUE(
      (std::is_same_v<
          typename traits::host_mirror_backend,
          typename Morpheus::HostMirror<typename traits::backend>::backend>));

  if (std::is_same_v<typename traits::memory_space, Kokkos::HostSpace>) {
    EXPECT_TRUE(traits::is_hostspace);
  } else {
    EXPECT_FALSE(traits::is_hostspace);
  }
}

TYPED_TEST(ContainerTraitsUnaryTest, TestLayout) {
  using traits = typename TestFixture::container_traits;
  using unary  = typename TestFixture::type;

  if (Morpheus::is_default_v<typename unary::array_layout>) {
    EXPECT_TRUE(
        (std::is_same_v<typename traits::array_layout,
                        typename traits::execution_space::array_layout>));
  } else {
    EXPECT_TRUE((Morpheus::has_same_layout_v<traits, unary>));
  }
}

TYPED_TEST(ContainerTraitsUnaryTest, TestMemoryTraits) {
  using traits = typename TestFixture::container_traits;

  // // TODO: Enable memory traits in unary container
  // if (Morpheus::is_default_v<typename unary::memory_traits>) {
  //   EXPECT_TRUE((std::is_same_v<typename traits::memory_traits,
  //                               typename Kokkos::MemoryManaged>));
  // } else {
  //   EXPECT_TRUE((std::is_same_v<typename traits::memory_traits,
  //                               typename unary::memory_traits>));
  // }

  // if (std::is_same_v<typename traits::memory_traits,
  //                    typename Kokkos::MemoryManaged>) {
  //   EXPECT_TRUE(traits::is_managed);
  // } else {
  //   EXPECT_FALSE(traits::is_managed);
  // }

  EXPECT_TRUE((std::is_same_v<typename traits::memory_traits,
                              typename Kokkos::MemoryManaged>));
  EXPECT_TRUE(traits::is_managed);
}

TYPED_TEST(ContainerTraitsUnaryTest, TestType) {
  using traits        = typename TestFixture::container_traits;
  using value_type    = typename traits::value_type;
  using index_type    = typename traits::index_type;
  using array_layout  = typename traits::array_layout;
  using backend       = typename traits::backend;
  using memory_traits = typename traits::memory_traits;
  using ctype = TestContainer<value_type, index_type, array_layout, backend,
                              memory_traits>;
  EXPECT_TRUE((std::is_same_v<typename traits::type, ctype>));
  EXPECT_TRUE((std::is_same_v<typename traits::pointer, ctype*>));
  EXPECT_TRUE((std::is_same_v<typename traits::const_pointer, const ctype*>));
  EXPECT_TRUE((std::is_same_v<typename traits::reference, ctype&>));
  EXPECT_TRUE((std::is_same_v<typename traits::const_reference, const ctype&>));
}

TYPED_TEST(ContainerTraitsUnaryTest, TestHostMirror) {
  using traits = typename TestFixture::container_traits;

  using value_type   = typename traits::non_const_value_type;
  using index_type   = typename traits::non_const_index_type;
  using array_layout = typename traits::array_layout;
  using exe          = typename traits::host_mirror_backend::execution_space;
  using mem          = typename traits::host_mirror_backend::memory_space;
  using back         = typename traits::host_mirror_backend::backend;
  using dev          = Morpheus::Device<exe, mem, back>;

  EXPECT_TRUE(
      (std::is_same_v<typename traits::HostMirror,
                      TestContainer<value_type, index_type, array_layout, dev,
                                    typename Kokkos::MemoryManaged>>));
}

TEST(ContainerTraitsTest, TestConst) {
  {
    using const_container = TestContainer<const double, const int>;
    EXPECT_TRUE((std::is_same_v<typename const_container::non_const_value_type,
                                double>));
    EXPECT_TRUE(
        (std::is_same_v<typename const_container::non_const_index_type, int>));
  }

  {
    using container = TestContainer<double, int>;
    EXPECT_TRUE(
        (std::is_same_v<typename container::const_value_type, const double>));
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_CONTAINERTRAITS_HPP
