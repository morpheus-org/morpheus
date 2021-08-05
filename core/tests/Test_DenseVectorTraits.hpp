/**
 * Test_DenseVectorTraits.hpp
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
          typename IndexTypeRef, typename LayoutRef, typename SpaceRef,
          typename SizeTypeRef = IndexTypeRef>
void test_DenseVector_traits() {
  // Type Traits
  ::testing::StaticAssertTypeEq<typename Container::type, Container>();
  ::testing::StaticAssertTypeEq<typename Container::tag, Tag>();

  // Value Traits
  ::testing::StaticAssertTypeEq<typename Container::value_type, ValueTypeRef>();
  ::testing::StaticAssertTypeEq<typename Container::non_const_value_type,
                                ValueTypeRef>();
  ::testing::StaticAssertTypeEq<typename Container::size_type, SizeTypeRef>();
  ::testing::StaticAssertTypeEq<typename Container::index_type, IndexTypeRef>();
  ::testing::StaticAssertTypeEq<typename Container::non_const_index_type,
                                IndexTypeRef>();
  ::testing::StaticAssertTypeEq<typename Container::array_layout, LayoutRef>();

  // Space Traits
  ::testing::StaticAssertTypeEq<typename Container::memory_space,
                                typename SpaceRef::memory_space>();
  ::testing::StaticAssertTypeEq<typename Container::execution_space,
                                typename SpaceRef::execution_space>();
  ::testing::StaticAssertTypeEq<
      typename Container::device_type,
      Kokkos::Device<typename Container::execution_space,
                     typename Container::memory_space>>();

  //  HostMirror Traits
  ::testing::StaticAssertTypeEq<
      typename Container::HostMirror,
      TEST_CONTAINER<typename Container::non_const_value_type,
                     typename Container::non_const_index_type,
                     typename Container::array_layout,
                     Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                                    typename Container::host_mirror_space::
                                        memory_space>>>();
  ::testing::StaticAssertTypeEq<
      typename Container::host_mirror_type,
      TEST_CONTAINER<typename Container::non_const_value_type,
                     typename Container::non_const_index_type,
                     typename Container::array_layout,
                     typename Container::host_mirror_space::memory_space>>();

  // Pointer Traits
  ::testing::StaticAssertTypeEq<typename Container::pointer,
                                typename std::add_pointer<Container>::type>();
  ::testing::StaticAssertTypeEq<
      typename Container::const_pointer,
      typename std::add_pointer<
          typename std::add_const<Container>::type>::type>();

  //   Reference Traits
  ::testing::StaticAssertTypeEq<
      typename Container::reference,
      typename std::add_lvalue_reference<Container>::type>();
  ::testing::StaticAssertTypeEq<
      typename Container::const_reference,
      typename std::add_lvalue_reference<
          typename std::add_const<Container>::type>::type>();
}

template <typename Container>
void test_DenseVector_format_traits() {
  ::testing::StaticAssertTypeEq<
      typename Container::value_array_pointer,
      typename std::add_pointer<typename Container::value_type>::type>();
  ::testing::StaticAssertTypeEq<typename Container::value_array_reference,
                                typename std::add_lvalue_reference<
                                    typename Container::value_type>::type>();
}

namespace Test {
TEST(TESTSUITE_NAME, DenseVectorTraits) {
  // DenseVectorTraits_dlr
  {
    using container =
        Morpheus::DenseVector<double, long long, Kokkos::LayoutRight,
                              TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, double, long long,
                            Kokkos::LayoutRight, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_dll
  {
    using container = Morpheus::DenseVector<double, long long,
                                            Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, double, long long,
                            Kokkos::LayoutLeft, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_dir
  {
    using container =
        Morpheus::DenseVector<double, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, double, int,
                            Kokkos::LayoutRight, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_dil
  {
    using container =
        Morpheus::DenseVector<double, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, double, int,
                            Kokkos::LayoutLeft, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_flr
  {
    using container =
        Morpheus::DenseVector<float, long long, Kokkos::LayoutRight,
                              TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, long long,
                            Kokkos::LayoutRight, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_fll
  {
    using container = Morpheus::DenseVector<float, long long,
                                            Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, long long,
                            Kokkos::LayoutLeft, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_fir
  {
    using container =
        Morpheus::DenseVector<float, int, Kokkos::LayoutRight, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, int,
                            Kokkos::LayoutRight, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_fil
  {
    using container =
        Morpheus::DenseVector<float, int, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, int,
                            Kokkos::LayoutLeft, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_f___
  {
    using container = Morpheus::DenseVector<float>;

    test_DenseVector_traits<
        container, TEST_CONTAINER_TAG, float, int,
        typename Kokkos::DefaultExecutionSpace::array_layout,
        Kokkos::DefaultExecutionSpace, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_f_l_
  {
    using container = Morpheus::DenseVector<float, Kokkos::LayoutLeft>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, int,
                            typename Kokkos::LayoutLeft,
                            Kokkos::DefaultExecutionSpace, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_fl_s
  {
    using container = Morpheus::DenseVector<float, long long, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, long long,
                            typename TEST_EXECSPACE::array_layout,
                            TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_fil_
  {
    using container =
        Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, long long,
                            Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace,
                            size_t>();

    test_DenseVector_format_traits<container>();
  }

  // DenseVectorTraits_f_ls
  {
    using container =
        Morpheus::DenseVector<float, Kokkos::LayoutLeft, TEST_EXECSPACE>;

    test_DenseVector_traits<container, TEST_CONTAINER_TAG, float, int,
                            Kokkos::LayoutLeft, TEST_EXECSPACE, size_t>();

    test_DenseVector_format_traits<container>();
  }
}

}  // namespace Test