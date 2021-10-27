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

#ifndef TEST_CORE_TEST_DENSEVECTOR_HPP
#define TEST_CORE_TEST_DENSEVECTOR_HPP

#include <Morpheus_Core.hpp>
// CMake Adds:
// #include <setup/Backends?>
#include <setup/TypeDefinition_Utils.hpp>

// using DenseVectorImplementations = ::testing::Types<
//     DenseVectorTypes<double, int, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<float, int, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<int, int, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<long long, int, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<double, long long, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<float, long long, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<int, long long, Kokkos::LayoutRight, Kokkos::Serial>,
//     DenseVectorTypes<long long, long long, Kokkos::LayoutRight,
//     Kokkos::Serial>, DenseVectorTypes<double, int, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<float, int, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<int, int, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<long long, int, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<double, long long, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<float, long long, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<int, long long, Kokkos::LayoutLeft,
//     Kokkos::Serial>, DenseVectorTypes<long long, long long,
//     Kokkos::LayoutLeft, Kokkos::Serial>>;

using DenseVectorImplementations = ::testing::Types<
    DenseVectorTypes<double, int, Kokkos::LayoutRight, Kokkos::Serial>>;

template <typename DenseVectorImplementations>
class DenseVectorTest : public ::testing::Test {
 public:
  using DenseVector = MorpheusContainers::DenseVector;
  using HostMirror  = typename MorpheusContainers::DenseVector::HostMirror;
  // No need for setup and tear-down in this case, mainly care about the types
  // any setup and tear-down will be made by each individual test
};

// TODO: Create similar class for when using different algorithms
// and setup a small, medium and large case vectors with assigned values
// do that in a separate setup file such that it is visible by different
// algorithms

namespace Test {

TYPED_TEST_CASE(DenseVectorTest, ContainerImplementations);

TYPED_TEST(DenseVectorTest, Traits) {
  // Check DenseVector Specific Traits:
  // Tag, value_array_type, value_array_pointer, value_array_reference
  // Repeat that for the HostMirror too
  // Check value_array_type traits too
  // Ensure size is of type size_t and not index_type
  // Add size_type trait
  static_assert(std::is_same<typename TestFixture::DenseVector::tag,
                             Morpheus::DenseVectorTag>::value);

  FAIL();
}

TYPED_TEST(DenseVectorTest, DefaultConstruction) {
  // DenseVector()

  FAIL();
}

TYPED_TEST(DenseVectorTest, NormalConstruction) {
  // DenseVector(const std::string name, index_type n, value_type val = 0)
  // DenseVector(index_type n, value_type val = 0)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorTest, RandomConstruction) {
  // DenseVector(const std::string name, index_type n, Generator rand_pool,
  //             const value_type range_low, const value_type range_high)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorTest, ShallowCopySemantics) {
  // DenseVector(const DenseVector<VR, PR...>& src, ...)
  // operator=(const DenseVector<VR, PR...>& src)
  // DenseVector(const DenseVector&) = default;
  // DenseVector& operator=(const DenseVector&) = default;

  FAIL();
}

TYPED_TEST(DenseVectorTest, Allocate) {
  // DenseVector& allocate(const std::string name,
  //                       const DenseVector<VR, PR...>& src)

  FAIL();
}

TYPED_TEST(DenseVectorTest, Assign) {
  // assign(const index_type n, const value_type val)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorTest, AssignRandom) {
  // assign(Generator rand_pool, const value_type range_low,
  //        const value_type range_high)

  FAIL();
}

TYPED_TEST(DenseVectorTest, Resize) {
  // resize(index_type n)
  // resize(const index_type n, const index_type val)
  // TODO: Change n to size_t

  FAIL();
}

TYPED_TEST(DenseVectorTest, UtilRoutines) {
  // size()
  // data()
  // view()
  // const_view()
  // name()

  FAIL();
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_HPP