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

TEST(TESTSUITE_NAME, DeepCopy_DenseVector_SameSpace_Mirror) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;
  vector x(10, -2), x_res(10, -10);
  // Always allocates new memory space
  auto x_mirror = Morpheus::create_mirror(x);

  Morpheus::copy(x, x_mirror);  // Space-Host
  Morpheus::copy(x_res, x);     // Should be shallow copy

  ASSERT_EQ(x_mirror.size(), x.size());
  for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], -2) << "Value of the mirror should be the same as "
                                  "the value of x during deep copy (-2)";
  }
  if (Morpheus::is_Host_Memoryspace_v<typename vector::memory_space>) {
    for (typename vector::index_type i = 0; i < x.size(); i++) {
      ASSERT_EQ(x[i], -10) << "x should be a shallow copy of x_res i.e value "
                              "of x[i]=x_res[i]=-10";
    }
  }
}

TEST(TESTSUITE_NAME, DeepCopy_DenseVector_SameSpace_MirrorContainer) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       TEST_EXECSPACE>;
  vector x(10, -2), x_res(10, -10);

  // Might perform shallow copy if already on host
  auto x_mirror = Morpheus::create_mirror_container(x);
  using mirror  = decltype(x_mirror);

  // if on host x_mirror should be a shallow copy of x
  Morpheus::copy(x, x_mirror);
  Morpheus::copy(x_res, x);

  ASSERT_EQ(x_mirror.size(), x_res.size());
  if (Morpheus::is_Host_Memoryspace_v<typename vector::memory_space>) {
    for (typename mirror::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], x_res[i])
          << "Value of the mirror should be the same as x_res (-10) as this is "
             "a shallow copy";
    }
  } else {
    for (typename vector::index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -2) << "Value of the mirror should be the same as "
                                    "the value of x during deep copy (-2)";
    }
  }
}

TEST(TESTSUITE_NAME, DeepCopy_DenseVector_DeviceHost) {
  using vector =
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft, Kokkos::Cuda>;
  vector x(10, -2);
  auto y = Morpheus::create_mirror(x);

  Morpheus::copy(x, y);  // DtoH

  ASSERT_EQ(y.size(), x.size());
  for (typename vector::index_type i = 0; i < y.size(); i++) {
    ASSERT_EQ(y[i], -2)
        << "Value of the mirror should be the same as the device vector";
  }
}

TEST(TESTSUITE_NAME, DeepCopy_DenseVector_HostDevice) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       Kokkos::HostSpace>;
  vector x(10, -2);
  auto y    = Morpheus::create_mirror<Kokkos::Cuda>(x);
  auto xres = Morpheus::create_mirror(y);

  Morpheus::copy(x, y);     // HtoD
  Morpheus::copy(y, xres);  // DtoH

  ASSERT_EQ(xres.size(), x.size());
  for (typename vector::index_type i = 0; i < xres.size(); i++) {
    x[i] = -5;
    ASSERT_EQ(xres[i], -2) << "Value of the mirror should be the same as the "
                              "initial value of x (-2)";
  }
}

TEST(TESTSUITE_NAME, DeepCopy_DenseVector_DeviceDevice) {
  using vector = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                       Kokkos::HostSpace>;
  vector x1(10, -2), x2(10, -4);
  auto y    = Morpheus::create_mirror<Kokkos::Cuda>(x1);
  auto z    = Morpheus::create_mirror<Kokkos::Cuda>(x2);
  auto xres = Morpheus::create_mirror(y);

  Morpheus::copy(x1, y);    // HtoD
  Morpheus::copy(x2, z);    // HtoD
  Morpheus::copy(z, y);     // DtoD
  Morpheus::copy(y, xres);  // DtoH

  ASSERT_EQ(xres.size(), x1.size());
  for (typename vector::index_type i = 0; i < xres.size(); i++) {
    x1[i] = -5;
    ASSERT_EQ(xres[i], -4) << "Value of the mirror should be the same as the "
                              "initial value of x2 (-4)";
  }
}

}  // namespace Test