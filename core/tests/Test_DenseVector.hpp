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
#include <utils/Utils.hpp>

using DenseVectorTypes =
    typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                               types::types_set>::type;
using DenseVectorUnary = to_gtest_types<DenseVectorTypes>::type;

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DenseVectorUnaryTest : public ::testing::Test {
 public:
  using type   = UnaryContainer;
  using device = typename UnaryContainer::type;
  using host   = typename UnaryContainer::type::HostMirror;
};

namespace Test {
/**
 * @brief Test Suite using the Unary DenseVectors
 *
 */
TYPED_TEST_SUITE(DenseVectorUnaryTest, DenseVectorUnary);

/**
 * @brief Testing default construction of DenseVector container
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultConstruction) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;

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
 * @brief Testing default copy assignment of DenseVector container from another
 * DenseVector container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultCopyAssignment) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size = 10;
  Vector x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector xh(x.size(), 0);
  Morpheus::copy(x, xh);

  HostVector yh = xh;
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector y = x;
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_type i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

/**
 * @brief Testing default copy construction of DenseVector container from
 * another DenseVector container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultCopyConstructor) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size = 10;
  Vector x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector xh(x.size(), (value_type)0);
  Morpheus::copy(x, xh);

  HostVector yh(xh);
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector y(x);
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_type i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

/**
 * @brief Testing default move assignment of DenseVector container from another
 * DenseVector container with the same parameters. Resulting container should be
 * a shallow copy of the original.
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultMoveAssignment) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size = 10;
  Vector x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector xh(x.size(), 0);
  Morpheus::copy(x, xh);

  HostVector yh = std::move(xh);
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector y = std::move(x);
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_type i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

/**
 * @brief Testing default move construction of DenseVector container from
 * another DenseVector container with the same parameters. Resulting container
 * should be a shallow copy of the original.
 *
 */
TYPED_TEST(DenseVectorUnaryTest, DefaultMoveConstructor) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size = 10;
  Vector x(size, (value_type)5.22);
  EXPECT_EQ(x.size(), size);

  HostVector xh(x.size(), 0);
  Morpheus::copy(x, xh);

  HostVector yh(std::move(xh));
  EXPECT_EQ(yh.size(), xh.size());

  xh[4] = (value_type)-4.33;
  xh[9] = (value_type)-9.44;

  yh[0] = 0;
  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], yh[i]);
  }

  // Now check device vector
  Vector y(std::move(x));
  EXPECT_EQ(y.size(), x.size());
  Morpheus::copy(xh, x);

  // Send other vector back to host for check
  HostVector yt(y.size(), 0);
  Morpheus::copy(y, yt);
  for (size_type i = 0; i < y.size(); i++) {
    EXPECT_EQ(yt[i], xh[i]);
  }
}

/**
 * @brief Testing construction of DenseVector container with size `n` and values
 * set at 0 by default
 *
 */
TYPED_TEST(DenseVectorUnaryTest, NormalConstructionDefaultVal) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size      = 100;
  value_type val = 0;

  Vector x(size);
  HostVector xh(size);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  for (size_type i = 0; i < x.size(); i++) {
    EXPECT_EQ(xh.data()[i], val);
  }
}

/**
 * @brief Testing construction of DenseVector container with size `n` and values
 * set to val
 *
 */
TYPED_TEST(DenseVectorUnaryTest, NormalConstruction) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size      = 100;
  value_type val = 15.22;

  Vector x(size, val);
  HostVector xh(size, val);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  for (size_type i = 0; i < x.size(); i++) {
    EXPECT_EQ(xh.data()[i], val);
  }
}

// /**
//  * @brief Testing construction of DenseVector from a raw pointer
//  *
//  */
// TYPED_TEST(DenseVectorUnaryTest, PointerConstruction) {
//   // DenseVector(size_type n, value_type val)
//   using Vector     = typename TestFixture::device;
//   using HostVector = typename TestFixture::host;
//   using size_type = typename Vector::size_type;
//   using value_type = typename Vector::value_type;

//   size_type size = 10000;
//   value_type val  = 15;

//   // Vector x(size, val);

//   value_type* xptr = (value_type*)malloc(size * sizeof(value_type));
//   for (size_type i = 0; i < size; i++) {
//     xptr[i] = (value_type)i;
//   }

//   {
//     HostVector xh(size, xptr);
//     EXPECT_EQ(xh.size(), size);

//     // Update contents of the vector
//     for (size_type i = 0; i < size; i++) {
//       xh[i] = val;
//     }
//   }

//   // xptr allocation should still exist here with the updated values
//   for (size_type i = 0; i < size; i++) {
//     EXPECT_EQ(xptr[i], val);
//   }

//   free(xptr);
// }

TYPED_TEST(DenseVectorUnaryTest, RandomConstruction) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size            = 10000;
  value_type low_bound = -5.0, high_bound = 25.0;
  unsigned long long seed = 5374857;
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);

  Vector x(size, rand_pool, low_bound, high_bound);
  HostVector xh(size, 0);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);

  // Check if empty
  int nzeros = 0;
  for (size_type i = 0; i < xh.size(); i++) {
    if (xh[i] == 0.0) {
      nzeros++;
    }
  }
  EXPECT_LT(nzeros, xh.size());

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_GE(xh[i], low_bound);
    EXPECT_LT(xh[i], high_bound);
  }
}

TYPED_TEST(DenseVectorUnaryTest, AssignNoResize) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto size = 10000;

  Vector x(size, 0);
  HostVector xh(size, 0);

  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  x.assign(100, (value_type)35.22);
  // Make sure we are not resizing if assignment_size < size
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < 100; i++) {
    EXPECT_EQ(xh[i], (value_type)35.22);
  }
  for (size_type i = 100; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(1000, (value_type)20.33);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < 1000; i++) {
    EXPECT_EQ(xh[i], (value_type)20.33);
  }
  for (size_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(80, (value_type)-30.11);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < 80; i++) {
    EXPECT_EQ(xh[i], (value_type)-30.11);
  }
  for (size_type i = 80; i < 1000; i++) {
    EXPECT_EQ(xh[i], (value_type)20.33);
  }
  for (size_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(size, (value_type)-1.111);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], (value_type)-1.111);
  }
}

// TYPED_TEST(DenseVectorUnaryTest, AssignResize) {
//   using Vector     = typename TestFixture::device;
//   using HostVector = typename TestFixture::host;
//   using size_type  = typename Vector::size_type;
//   using value_type = typename Vector::value_type;

//   auto size = 10000;
//   Vector x(size, 0);

//   // x should resize now that the size we are assigning is larger that `size`
//   x.assign(size + 2000, (value_type)10.111);
//   EXPECT_EQ(x.size(), size + 2000);

//   HostVector xh(x.size(), 0);
//   EXPECT_EQ(xh.size(), x.size());
//   Morpheus::copy(x, xh);

//   for (size_type i = 0; i < xh.size(); i++) {
//     EXPECT_EQ(xh[i], (value_type)10.111);
//   }
// }

TYPED_TEST(DenseVectorUnaryTest, AssignRandomNoResize) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;

  auto size               = 10000;
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

  for (size_type i = 0; i < 100; i++) {
    EXPECT_GE(xh[i], 10);
    EXPECT_LT(xh[i], 30);
  }
  for (size_type i = 100; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  //
  x.assign(1000, rand_pool, 40, 50);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < 1000; i++) {
    EXPECT_GE(xh[i], 40);
    EXPECT_LT(xh[i], 50);
  }
  for (size_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(80, rand_pool, -4, 5);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < 80; i++) {
    EXPECT_GE(xh[i], -4);
    EXPECT_LT(xh[i], 5);
  }
  for (size_type i = 80; i < 1000; i++) {
    EXPECT_GE(xh[i], 40);
    EXPECT_LT(xh[i], 50);
  }
  for (size_type i = 1000; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }

  x.assign(size, rand_pool, 60, 70);
  EXPECT_EQ(x.size(), size);
  Morpheus::copy(x, xh);

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_GE(xh[i], 60);
    EXPECT_LT(xh[i], 70);
  }
}

// TYPED_TEST(DenseVectorUnaryTest, AssignRandomResize) {
//   using Vector     = typename TestFixture::device;
//   using HostVector = typename TestFixture::host;
//   using size_type  = typename Vector::size_type;

//   auto size = 10000, seed = 5374857;
//   Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> rand_pool(seed);
//   Vector x(size, 0);

//   // x should resize now that the size we are assigning is larger that `size`
//   x.assign(size + 2000, rand_pool, 30, 40);
//   EXPECT_EQ(x.size(), size + 2000);

//   HostVector xh(x.size(), 0);
//   EXPECT_EQ(xh.size(), x.size());
//   Morpheus::copy(x, xh);

//   for (size_type i = 0; i < xh.size(); i++) {
//     EXPECT_GE(xh[i], 30);
//     EXPECT_LT(xh[i], 40);
//   }
// }

TYPED_TEST(DenseVectorUnaryTest, Resize) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto original_size            = 1000;
  const value_type original_val = 10.11;

  Vector x(original_size, original_val);
  HostVector xh(original_size, 0);
  EXPECT_EQ(x.size(), original_size);
  EXPECT_EQ(xh.size(), original_size);

  size_type smaller_size = 100;
  x.resize(smaller_size);
  EXPECT_EQ(x.size(), smaller_size);

  xh.resize(smaller_size);
  EXPECT_EQ(xh.size(), smaller_size);

  Morpheus::copy(x, xh);

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], original_val);
  }

  auto larger_size = 10000;

  x.resize(larger_size);
  EXPECT_EQ(x.size(), larger_size);
  xh.resize(larger_size);
  EXPECT_EQ(xh.size(), larger_size);

  Morpheus::copy(x, xh);
  // Values from smaller_size onwards should be set to zero
  for (size_type i = 0; i < smaller_size; i++) {
    EXPECT_EQ(xh[i], original_val);
  }
  for (size_type i = smaller_size; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], 0.0);
  }
}

TYPED_TEST(DenseVectorUnaryTest, ResizeVal) {
  using Vector     = typename TestFixture::device;
  using HostVector = typename TestFixture::host;
  using size_type  = typename Vector::size_type;
  using value_type = typename Vector::value_type;

  auto original_size            = 1000;
  const value_type original_val = 10.11;

  Vector x(original_size, original_val);
  HostVector xh(original_size, 0);
  EXPECT_EQ(x.size(), original_size);
  EXPECT_EQ(xh.size(), original_size);

  auto smaller_size            = 100;
  const value_type smaller_val = 2.22;

  x.resize(smaller_size, smaller_val);
  EXPECT_EQ(x.size(), smaller_size);

  xh.resize(smaller_size, 0);
  EXPECT_EQ(xh.size(), smaller_size);

  Morpheus::copy(x, xh);

  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], smaller_val);
  }

  auto larger_size            = 10000;
  const value_type larger_val = 33.33;

  x.resize(larger_size, larger_val);
  EXPECT_EQ(x.size(), larger_size);
  xh.resize(larger_size, 0);
  EXPECT_EQ(xh.size(), larger_size);

  Morpheus::copy(x, xh);
  // Values from smaller_size onwards should be set to zero
  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], larger_val);
  }
}

TYPED_TEST(DenseVectorUnaryTest, ElementAccess) {
  using Vector              = typename TestFixture::device;
  using HostVector          = typename TestFixture::host;
  using size_type           = typename Vector::size_type;
  using value_type          = typename Vector::value_type;
  using value_array_pointer = typename Vector::value_array_pointer;

  auto size            = 50;
  const value_type val = (value_type)10.11;

  Vector x(size, val);
  HostVector xh(size, 0);
  EXPECT_EQ(x.size(), size);
  EXPECT_EQ(xh.size(), size);

  Morpheus::copy(x, xh);
  value_array_pointer xptr = xh.data();

  xptr[2]  = (value_type)2.22;
  xptr[15] = (value_type)15.15;
  // Values from smaller_size onwards should be set to zero
  for (size_type i = 0; i < xh.size(); i++) {
    EXPECT_EQ(xh[i], xh(i));
    if (i == 2) {
      EXPECT_EQ(xptr[i], (value_type)2.22);
    } else if (i == 15) {
      EXPECT_EQ(xptr[i], (value_type)15.15);
    } else {
      EXPECT_EQ(xh[i], xh(i));
    }
  }
}

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEVECTOR_HPP
