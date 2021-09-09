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
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_SameSpace_Mirror) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container  = Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using index_type = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);
  container Ar("DenseMatrix", 4, 3, 2);

  // Always allocates new memory space on host
  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // Space-Host
  Morpheus::copy(Ar, A);        // Should always be shallow copy

  check_shapes(A, A_mirror, Morpheus::DenseMatrixTag{});
  check_shapes(Ar, A_mirror, Morpheus::DenseMatrixTag{});

  for (index_type i = 0; i < A_mirror.nrows(); i++) {
    for (index_type j = 0; j < A_mirror.ncols(); j++) {
      ASSERT_EQ(A_mirror(i, j), 1)
          << "Value of the mirror values should be the same as the value of "
             "A.values during deep copy (5)";
    }
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
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_SameSpace_MirrorContainer) {
  using test_memory_space = typename TEST_EXECSPACE::memory_space;
  using container  = Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                                          test_memory_space>;
  using index_type = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);
  container Ar("DenseMatrix", 4, 3, 2);

  // Might perform shallow copy if already on host
  auto A_mirror = Morpheus::create_mirror_container(A);

  // if on host x_mirror should be a shallow copy of x
  Morpheus::copy(A, A_mirror);
  Morpheus::copy(Ar, A);

  check_shapes(A, A_mirror, Morpheus::DenseMatrixTag{});
  check_shapes(Ar, A_mirror, Morpheus::DenseMatrixTag{});

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space>) {
    for (index_type i = 0; i < A_mirror.nrows(); i++) {
      for (index_type j = 0; j < A_mirror.ncols(); j++) {
        ASSERT_EQ(A_mirror(i, j), Ar(i, j))
            << "Value of the mirror values should be the same as the value of "
               "Ar.values due to Shallow Copy (2)";
      }
    }
  } else {
    for (index_type i = 0; i < A_mirror.nrows(); i++) {
      for (index_type j = 0; j < A_mirror.ncols(); j++) {
        ASSERT_EQ(A_mirror(i, j), 1)
            << "Value of the mirror values should be the same as the value of "
               "A.values during deep copy (1)";
      }
    }
  }
}

#if defined(MORPHEUS_ENABLE_CUDA)

// Creates a mirror on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_DeviceHost) {
  using container = Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                                          typename Kokkos::Cuda::memory_space>;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);

  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // DtoH

  check_shapes(A, A_mirror, Morpheus::DenseMatrixTag{});
  for (index_type i = 0; i < A_mirror.nrows(); i++) {
    for (index_type j = 0; j < A_mirror.ncols(); j++) {
      ASSERT_EQ(A_mirror(i, j), 1)
          << "Value of the mirror values should be the same as the value of "
             "A.values during deep copy (1)";
    }
  }
}

// Creates a mirror container on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_DeviceHost_MirrorCotnainer) {
  using container = Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                                          typename Kokkos::Cuda::memory_space>;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);

  auto A_mirror = Morpheus::create_mirror_container(A);

  Morpheus::copy(A, A_mirror);  // DtoH

  check_shapes(A, A_mirror, Morpheus::DenseMatrixTag{});
  for (index_type i = 0; i < A_mirror.nrows(); i++) {
    for (index_type j = 0; j < A_mirror.ncols(); j++) {
      ASSERT_EQ(A_mirror(i, j), 1)
          << "Value of the mirror values should be the same as the value of "
             "A.values during deep copy (1)";
    }
  }
}

// Creates a mirror on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_HostDevice) {
  using container =
      Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                            typename Kokkos::HostSpace::memory_space>;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);

  auto A_mirror_dev  = Morpheus::create_mirror<Kokkos::Cuda>(A);
  auto A_mirror_host = Morpheus::create_mirror(A_mirror_dev);

  Morpheus::copy(A, A_mirror_dev);              // HtoD
  Morpheus::copy(A_mirror_dev, A_mirror_host);  // DtoH

  check_shapes(A, A_mirror_host, Morpheus::DenseMatrixTag{});
  check_shapes(A, A_mirror_dev, Morpheus::DenseMatrixTag{});

  // Change the value to main container to check if we did shallow copy
  A.assign(A.nrows(), A.ncols(), 5);
  for (index_type i = 0; i < A_mirror_host.nrows(); i++) {
    for (index_type j = 0; j < A_mirror_host.ncols(); j++) {
      ASSERT_EQ(A_mirror_host(i, j), 1) << "Value of the mirror host values "
                                           "should be the same as the value of "
                                           "A.values during deep copy (1)";
    }
  }
}

// Creates a mirror container on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_HostDevice_MirrorContainer) {
  using container =
      Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                            typename Kokkos::HostSpace::memory_space>;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);

  auto A_mirror_dev  = Morpheus::create_mirror_container<Kokkos::Cuda>(A);
  auto A_mirror_host = Morpheus::create_mirror_container(A_mirror_dev);

  Morpheus::copy(A, A_mirror_dev);              // HtoD
  Morpheus::copy(A_mirror_dev, A_mirror_host);  // DtoH

  check_shapes(A, A_mirror_host, Morpheus::DenseMatrixTag{});
  check_shapes(A, A_mirror_dev, Morpheus::DenseMatrixTag{});

  // Change the value to main container to check if we did shallow copy
  A.assign(A.nrows(), A.ncols(), 5);
  for (index_type i = 0; i < A_mirror_host.nrows(); i++) {
    for (index_type j = 0; j < A_mirror_host.ncols(); j++) {
      ASSERT_EQ(A_mirror_host(i, j), 1) << "Value of the mirror host values "
                                           "should be the same as the value of "
                                           "A.values during deep copy (1)";
    }
  }
}

// Creates two mirror containers on device from host
// Issues a copy between host to device and then a copy between
// the two device mirrors. Then sends the result back to host
// to be compared which should match the initial state of Ar.
TEST(TESTSUITE_NAME, DeepCopy_DenseMatrix_DeviecDevice_MirrorContainer) {
  using container =
      Morpheus::DenseMatrix<float, long long, Kokkos::LayoutLeft,
                            typename Kokkos::HostSpace::memory_space>;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  container A("DenseMatrix", 4, 3, 1);
  container Ar("DenseMatrix", 4, 3, 2);

  auto A_mirror_dev1 = Morpheus::create_mirror_container<Kokkos::Cuda>(A);
  auto A_mirror_dev2 = Morpheus::create_mirror_container<Kokkos::Cuda>(Ar);
  auto Ares          = Morpheus::create_mirror(A_mirror_dev1);

  Morpheus::copy(A, A_mirror_dev1);              // HtoD
  Morpheus::copy(Ar, A_mirror_dev2);             // HtoD
  Morpheus::copy(A_mirror_dev2, A_mirror_dev1);  // DtoD
  Morpheus::copy(A_mirror_dev1, Ares);           // DtoH

  check_shapes(A, Ares, Morpheus::DenseMatrixTag{});
  // Change the value to main container to check if we did shallow copy
  A.assign(A.nrows(), A.ncols(), 3);
  Ar.assign(Ar.nrows(), Ar.ncols(), 3);
  for (index_type i = 0; i < Ares.nrows(); i++) {
    for (index_type j = 0; j < Ares.ncols(); j++) {
      ASSERT_EQ(Ares(i, j), 2) << "Value of the Ares should be the same as the "
                                  "initial value of Ar during deep copy (2)";
    }
  }
}

#endif  // MORPHEUS_ENABLE_CUDA

}  // namespace Test