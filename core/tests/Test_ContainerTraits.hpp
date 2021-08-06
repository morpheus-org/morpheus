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

#include <Morpheus_Core.hpp>

template <typename Container, typename Tag, typename ValueTypeRef,
          typename IndexTypeRef, typename LayoutRef, typename SpaceRef>
void test_container_traits() {
  // Type Traits
  static_assert(std::is_same<typename Container::type, Container>::value);
  static_assert(std::is_same<typename Container::tag, Tag>::value);

  // Value Traits
  static_assert(
      std::is_same<typename Container::value_type, ValueTypeRef>::value);
  static_assert(std::is_same<typename Container::non_const_value_type,
                             ValueTypeRef>::value);
  static_assert(
      std::is_same<typename Container::index_type, IndexTypeRef>::value);
  static_assert(std::is_same<typename Container::non_const_index_type,
                             IndexTypeRef>::value);
  static_assert(
      std::is_same<typename Container::array_layout, LayoutRef>::value);

  // Space Traits
  static_assert(std::is_same<typename Container::memory_space,
                             typename SpaceRef::memory_space>::value);
  static_assert(std::is_same<typename Container::execution_space,
                             typename SpaceRef::execution_space>::value);
  static_assert(
      std::is_same<typename Container::device_type,
                   Kokkos::Device<typename Container::execution_space,
                                  typename Container::memory_space>>::value);

  //  HostMirror Traits
  static_assert(
      std::is_same<
          typename Container::HostMirror,
          TEST_CONTAINER<typename Container::non_const_value_type,
                         typename Container::non_const_index_type,
                         typename Container::array_layout,
                         Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                                        typename Container::host_mirror_space::
                                            memory_space>>>::value);
  static_assert(
      std::is_same<
          typename Container::host_mirror_type,
          TEST_CONTAINER<typename Container::non_const_value_type,
                         typename Container::non_const_index_type,
                         typename Container::array_layout,
                         typename Container::host_mirror_space::memory_space>>::
          value);

  // Pointer Traits
  static_assert(
      std::is_same<typename Container::pointer,
                   typename std::add_pointer<Container>::type>::value);
  static_assert(
      std::is_same<typename Container::const_pointer,
                   typename std::add_pointer<
                       typename std::add_const<Container>::type>::type>::value);

  //   Reference Traits
  static_assert(
      std::is_same<typename Container::reference,
                   typename std::add_lvalue_reference<Container>::type>::value);
  static_assert(
      std::is_same<typename Container::const_reference,
                   typename std::add_lvalue_reference<
                       typename std::add_const<Container>::type>::type>::value);
}

template <typename Container, typename Tag, typename ValueTypeRef,
          typename IndexTypeRef, typename LayoutRef, typename SpaceRef>
void test_traits() {
  test_container_traits<Container, Tag, ValueTypeRef, IndexTypeRef, LayoutRef,
                        SpaceRef>();

  test_traits<Container>(Tag());
}

namespace Test {
TEST(TESTSUITE_NAME, ContainersTraits) {
  // ContainerTraits_dlr
  {
    using container =
        TEST_CONTAINER<double, long long, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, double, long long,
                Kokkos::LayoutRight, TEST_EXECSPACE>();
  }

  // ContainerTraits_dll
  {
    using container =
        TEST_CONTAINER<double, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, double, long long,
                Kokkos::LayoutLeft, TEST_EXECSPACE>();
  }

  // ContainerTraits_dir
  {
    using container =
        TEST_CONTAINER<double, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, double, int, Kokkos::LayoutRight,
                TEST_EXECSPACE>();
  }

  // ContainerTraits_dil
  {
    using container =
        TEST_CONTAINER<double, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, double, int, Kokkos::LayoutLeft,
                TEST_EXECSPACE>();
  }

  // ContainerTraits_flr
  {
    using container =
        TEST_CONTAINER<float, long long, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, long long,
                Kokkos::LayoutRight, TEST_EXECSPACE>();
  }

  // ContainerTraits_fll
  {
    using container =
        TEST_CONTAINER<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, long long,
                Kokkos::LayoutLeft, TEST_EXECSPACE>();
  }

  // ContainerTraits_fir
  {
    using container =
        TEST_CONTAINER<float, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, int, Kokkos::LayoutRight,
                TEST_EXECSPACE>();
  }

  // ContainerTraits_fil
  {
    using container =
        TEST_CONTAINER<float, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, int, Kokkos::LayoutLeft,
                TEST_EXECSPACE>();
  }

  // ContainerTraits_f___
  {
    using container = TEST_CONTAINER<float>;

    test_traits<container, TEST_CONTAINER_TAG, float, int,
                typename Kokkos::DefaultExecutionSpace::array_layout,
                Kokkos::DefaultExecutionSpace>();
  }

  // ContainerTraits_f_l_
  {
    using container = TEST_CONTAINER<float, Kokkos::LayoutLeft>;

    test_traits<container, TEST_CONTAINER_TAG, float, int,
                typename Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>();
  }

  // ContainerTraits_fl_s
  {
    using container = TEST_CONTAINER<float, long long, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, long long,
                typename TEST_EXECSPACE::array_layout, TEST_EXECSPACE>();
  }

  // ContainerTraits_fil_
  {
    using container = TEST_CONTAINER<float, long long, Kokkos::LayoutLeft>;

    test_traits<container, TEST_CONTAINER_TAG, float, long long,
                Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>();
  }

  // Container_f_ls
  {
    using container = TEST_CONTAINER<float, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_traits<container, TEST_CONTAINER_TAG, float, int, Kokkos::LayoutLeft,
                TEST_EXECSPACE>();
  }
}

}  // namespace Test