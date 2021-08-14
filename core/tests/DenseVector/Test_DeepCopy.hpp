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

// Creates a mirror on host
// Issues a copy between space and host which is always deep
// Therefore checking the mirror container, it should maintain it's own
// state.
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_SameSpace_Mirror) {
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          TEST_EXECSPACE>;
  using index_type = typename container::index_type;

  container x(10, -2), x_res(10, -10);
  // Always allocates new memory space
  auto x_mirror = Morpheus::create_mirror(x);

  Morpheus::copy(x, x_mirror);  // Space-Host
  Morpheus::copy(x_res, x);     // Should be shallow copy

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});

  for (index_type i = 0; i < x_mirror.size(); i++) {
    ASSERT_EQ(x_mirror[i], -2) << "Value of the mirror should be the same as "
                                  "the value of x during deep copy (-2)";
  }
}

// Creates a mirror container on host
// Issues a copy between space and host
// If in different spaces, copy should be deep, otherwise shallow
// Then issues another copy which should always be shallow as the two
// containers are of the same type.
// Therefore checking the mirror container, it should maintain it's own
// state when the initial container lives in different state, otherwise
// it should have a shared state.
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_SameSpace_MirrorContainer) {
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          TEST_EXECSPACE>;
  using index_type = typename container::index_type;

  container x(10, -2), x_res(10, -10);

  // Might perform shallow copy if already on host
  auto x_mirror = Morpheus::create_mirror_container(x);
  using mirror  = decltype(x_mirror);

  // if on host x_mirror should be a shallow copy of x
  Morpheus::copy(x, x_mirror);
  Morpheus::copy(x_res, x);

  check_shapes(x, x_mirror, Morpheus::DenseVectorTag{});
  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space>) {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], x_res[i])
          << "Value of the mirror should be the same as x_res (-10) as this is "
             "a shallow copy";
    }
  } else {
    for (index_type i = 0; i < x_mirror.size(); i++) {
      ASSERT_EQ(x_mirror[i], -2) << "Value of the mirror should be the same as "
                                    "the value of x during deep copy (-2)";
    }
  }
}

#if defined(MORPHEUS_ENABLE_CUDA)

// Creates a mirror on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_DeviceHost) {
  using container =
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft, Kokkos::Cuda>;
  using index_type = typename container::index_type;

  container x(10, -2);
  auto y = Morpheus::create_mirror(x);

  Morpheus::copy(x, y);  // DtoH

  check_shapes(x, y, Morpheus::DenseVectorTag{});
  for (index_type i = 0; i < y.size(); i++) {
    ASSERT_EQ(y[i], -2)
        << "Value of the mirror should be the same as the device vector";
  }
}

// Creates a mirror container on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_DeviceHost_MirrorContainer) {
  using container =
      Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft, Kokkos::Cuda>;
  using index_type = typename container::index_type;

  container x(10, -2);
  auto y = Morpheus::create_mirror_container(x);

  Morpheus::copy(x, y);  // DtoH

  check_shapes(x, y, Morpheus::DenseVectorTag{});
  for (index_type i = 0; i < y.size(); i++) {
    ASSERT_EQ(y[i], -2)
        << "Value of the mirror should be the same as the device vector";
  }
}

// Creates a mirror on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_HostDevice) {
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          Kokkos::HostSpace>;
  using index_type = typename container::index_type;

  container x(10, -2);
  auto y    = Morpheus::create_mirror<Kokkos::Cuda>(x);
  auto xres = Morpheus::create_mirror(y);

  Morpheus::copy(x, y);     // HtoD
  Morpheus::copy(y, xres);  // DtoH

  check_shapes(x, xres, Morpheus::DenseVectorTag{});
  // Change the value to main container to check if we did shallow copy
  x.assign(x.size(), -5);
  for (index_type i = 0; i < xres.size(); i++) {
    ASSERT_EQ(xres[i], -2) << "Value of the mirror should be the same as the "
                              "initial value of x (-2)";
  }
}

// Creates a mirror container on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_HostDevice_MirrorContainer) {
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          Kokkos::HostSpace>;
  using index_type = typename container::index_type;

  container x(10, -2);
  auto y    = Morpheus::create_mirror_container<Kokkos::Cuda>(x);
  auto xres = Morpheus::create_mirror_container(y);

  Morpheus::copy(x, y);     // HtoD
  Morpheus::copy(y, xres);  // DtoH

  check_shapes(x, xres, Morpheus::DenseVectorTag{});
  // Change the value to main container to check if we did shallow copy
  x.assign(x.size(), -5);
  for (index_type i = 0; i < xres.size(); i++) {
    ASSERT_EQ(xres[i], -2) << "Value of the mirror should be the same as the "
                              "initial value of x (-2)";
  }
}

// Creates two mirror containers on device from host
// Issues a copy between host to device and then a copy between
// the two device mirrors. Then sends the result back to host
// to be compared which should match the initial state of x2.
TEST(TESTSUITE_NAME, DeepCopy_DenseVector_DeviceDevice) {
  using container  = Morpheus::DenseVector<float, long long, Kokkos::LayoutLeft,
                                          Kokkos::HostSpace>;
  using index_type = typename container::index_type;

  container x1(10, -2), x2(10, -4);
  auto y    = Morpheus::create_mirror<Kokkos::Cuda>(x1);
  auto z    = Morpheus::create_mirror<Kokkos::Cuda>(x2);
  auto xres = Morpheus::create_mirror(y);

  Morpheus::copy(x1, y);    // HtoD
  Morpheus::copy(x2, z);    // HtoD
  Morpheus::copy(z, y);     // DtoD
  Morpheus::copy(y, xres);  // DtoH

  check_shapes(x1, xres, Morpheus::DenseVectorTag{});
  // Change the value to main container to check if we did shallow copy
  x1.assign(x1.size(), -5);
  for (index_type i = 0; i < xres.size(); i++) {
    ASSERT_EQ(xres[i], -4) << "Value of the mirror should be the same as the "
                              "initial value of x2 (-4)";
  }
}

#endif  // MORPHEUS_ENABLE_CUDA

}  // namespace Test