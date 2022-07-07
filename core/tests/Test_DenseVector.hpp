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

#include <setup/DenseVectorDefinition_Utils.hpp>

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DenseVectorUnaryTest : public ::testing::Test {
 public:
  using DenseVector = UnaryContainer;
  using HostMirror  = typename UnaryContainer::HostMirror;
  // No need for setup and tear-down in this case, mainly care about the types
  // any setup and tear-down will be made by each individual test
};

// Used for testing behaviour between many types of the same container
template <typename DenseVectorTypes>
class DenseVectorTypesTest : public ::testing::Test {
 public:
  using DenseVector_v    = typename DenseVectorTypes::v;
  using DenseVector_vl   = typename DenseVectorTypes::vl;
  using DenseVector_vis  = typename DenseVectorTypes::vis;
  using DenseVector_vil  = typename DenseVectorTypes::vil;
  using DenseVector_vils = typename DenseVectorTypes::vils;
  using DenseVector_vls  = typename DenseVectorTypes::vls;
};

namespace Test {

TYPED_TEST_CASE(DenseVectorUnaryTest, DenseVectorUnary);

TYPED_TEST(DenseVectorUnaryTest, Traits) {
  // Check DenseVector Specific Traits:
  // Tag, value_array_type, value_array_pointer, value_array_reference
  // Repeat that for the HostMirror too
  // Check value_array_type traits too
  // Ensure size is of type size_t and not index_type
  // Add size_type trait
  static_assert(std::is_same<typename TestFixture::DenseVector::tag,
                             Morpheus::DenseVectorTag>::value);
}

/**
 * @brief Testing default construction of DenseVector container
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultConstruction) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;

  Vector x;
  EXPECT_EQ(x.size(), 0);
  EXPECT_EQ(x.data(), nullptr);
  EXPECT_EQ(x.view().size(), 0);
  EXPECT_EQ(x.view().data(), nullptr);

  HostVector xh;
  EXPECT_EQ(xh.size(), 0);
  EXPECT_EQ(xh.data(), nullptr);
  EXPECT_EQ(xh.view().size(), 0);
  EXPECT_EQ(xh.view().data(), nullptr);
}

/**
 * @brief Testing construction of DenseVector container with size `n` and values
 * set at 0 by default
 *
 */
TYPED_TEST(DenseVectorUnaryTest, NormalConstructionDefaultVal) {
  // DenseVector(index_type n, value_type val = 0)
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

  index_type size = 100;
  value_type val  = 0;

  Vector x(size);
  HostVector xh(size);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  for (index_type i = 0; i < x.size(); i++) {
    EXPECT_EQ(xh.data()[i], val);
  }
}

/**
 * @brief Testing construction of DenseVector container with size `n` and values
 * set at 0 by default
 *
 */
TYPED_TEST(DenseVectorUnaryTest, NormalConstruction) {
  // DenseVector(index_type n, value_type val)
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

  index_type size = 100;
  value_type val  = 15;

  Vector x(size, val);
  HostVector xh(size, val);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  for (index_type i = 0; i < x.size(); i++) {
    EXPECT_EQ(xh.data()[i], val);
  }
}

// /**
//  * @brief Testing construction of DenseVector from a raw pointer
//  *
//  */
// TYPED_TEST(DenseVectorUnaryTest, PointerConstruction) {
//   // DenseVector(index_type n, value_type val)
//   using Vector     = typename TestFixture::DenseVector;
//   using HostVector = typename TestFixture::HostMirror;
//   using index_type = typename Vector::index_type;
//   using value_type = typename Vector::value_type;

//   index_type size = 10000;
//   value_type val  = 15;

//   // Vector x(size, val);

//   value_type* xptr = (value_type*)malloc(size * sizeof(value_type));
//   for (index_type i = 0; i < size; i++) {
//     xptr[i] = (value_type)i;
//   }

//   {
//     HostVector xh(size, xptr);
//     EXPECT_EQ(xh.size(), size);

//     // Update contents of the vector
//     for (index_type i = 0; i < size; i++) {
//       xh[i] = val;
//     }
//   }

//   // xptr allocation should still exist here with the updated values
//   for (index_type i = 0; i < size; i++) {
//     EXPECT_EQ(xptr[i], val);
//   }

//   free(xptr);
// }

TYPED_TEST(DenseVectorUnaryTest, RandomConstruction) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;
  using value_type = typename Vector::value_type;

  index_type size      = 10000;
  value_type low_bound = -5.0, high_bound = 25.0;
  unsigned long long seed = 5374857;
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);

  Vector x(size, rand_pool, low_bound, high_bound);
  HostVector xh(size, 0);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  for (index_type i = 0; i < xh.size(); i++) {
    EXPECT_NE(xh[i], 0.0);
    EXPECT_GT(xh[i], low_bound);
    EXPECT_LT(xh[i], high_bound);
  }
}

// // Changed that to DenseVectorBinaryTests
// TYPED_TEST(DenseVectorUnaryTest, ShallowCopySemantics) {
//   // DenseVector(const DenseVector<VR, PR...>& src, ...)
//   // operator=(const DenseVector<VR, PR...>& src)
//   // DenseVector(const DenseVector&) = default;
//   // DenseVector& operator=(const DenseVector&) = default;
// }

// // Changed that to DenseVectorBinaryTests
// TYPED_TEST(DenseVectorUnaryTest, Allocate) {
//   // DenseVector& allocate(const std::string name,
//   //                       const DenseVector<VR, PR...>& src)
// }

TYPED_TEST(DenseVectorUnaryTest, AssignNoResize) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;

  index_type size = 10000;

  Vector x(size, 0);
  HostVector xh(size, 0);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  x.assign(100, 35);
  // Make sure we are not resizing if assignment_size < size
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 100; i++) {
    EXPECT_EQ(xh[i], 35.0);
  }
  for (index_type i = 100; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(1000, 20);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 1000; i++) {
    EXPECT_EQ(xh[i], 20.0);
  }
  for (index_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(80, -30);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 80; i++) {
    EXPECT_EQ(xh[i], -30.0);
  }
  for (index_type i = 80; i < 1000; i++) {
    EXPECT_EQ(xh[i], 20.0);
  }
  for (index_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(size, -1);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], -1.0);
  }
}

TYPED_TEST(DenseVectorUnaryTest, AssignResize) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;

  index_type size = 10000;
  Vector x(size, 0);

  // x should resize now that the size we are assigning is larger that `size`
  x.assign(size + 2000, 10);
  EXPECT_EQ(x.size(), size + 2000);

  HostVector xh(x.size(), 0);
  EXPECT_EQ(xh.size(), x.size());
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 10.0);
  }
}

TYPED_TEST(DenseVectorUnaryTest, AssignRandomNoResize) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;

  index_type size         = 10000;
  unsigned long long seed = 5374857;
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);

  Vector x(size, 0);
  HostVector xh(size, 0);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  x.assign(100, rand_pool, 10, 30);
  // Make sure we are not resizing if assignment_size < size
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 100; i++) {
    EXPECT_GT(xh[i], 10);
    EXPECT_LT(xh[i], 30);
  }
  for (index_type i = 100; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  //
  x.assign(1000, rand_pool, 40, 50);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 1000; i++) {
    EXPECT_GT(xh[i], 40);
    EXPECT_LT(xh[i], 50);
  }
  for (index_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(80, rand_pool, -4, 5);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < 80; i++) {
    EXPECT_GT(xh[i], -4);
    EXPECT_LT(xh[i], 5);
  }
  for (index_type i = 80; i < 1000; i++) {
    EXPECT_GT(xh[i], 40);
    EXPECT_LT(xh[i], 50);
  }
  for (index_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(size, rand_pool, 60, 70);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < xh.size(); i++) {
    EXPECT_GT(xh[i], 60);
    EXPECT_LT(xh[i], 70);
  }
}

TYPED_TEST(DenseVectorUnaryTest, AssignRandomResize) {
  using Vector     = typename TestFixture::DenseVector;
  using HostVector = typename TestFixture::HostMirror;
  using index_type = typename Vector::index_type;

  index_type size         = 10000;
  unsigned long long seed = 5374857;
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);
  Vector x(size, 0);

  // x should resize now that the size we are assigning is larger that `size`
  x.assign(size + 2000, rand_pool, 30, 40);
  EXPECT_EQ(x.size(), size + 2000);

  HostVector xh(x.size(), 0);
  EXPECT_EQ(xh.size(), x.size());
  Morpheus::copy(x, xh);

  for (index_type i = 0; i < xh.size(); i++) {
    EXPECT_GT(xh[i], 30);
    EXPECT_LT(xh[i], 40);
  }
}

// TYPED_TEST(DenseVectorUnaryTest, Resize) {
//   // resize(index_type n)
//   // resize(const index_type n, const index_type val)
//   // TODO: Change n to size_t
// }

// TYPED_TEST(DenseVectorUnaryTest, UtilRoutines) {
//   // size()
//   // data()
//   // view()
//   // const_view()
//   // name()
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_HPP
