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
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_SameSpace_Mirror) {
  using container =
      Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  index_array_type Ar_rind(5, 0), Ar_cind(2, 1);
  value_array_type Ar_val(2, -1);
  container Ar("Csr", 4, 3, 2, Ar_rind, Ar_cind, Ar_val);

  // Always allocates new memory space
  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // Space-Host
  Morpheus::copy(Ar, A);        // Should be shallow copy

  check_shapes(A, A_mirror, Morpheus::CsrTag{});
  for (index_type i = 0; i < A_mirror.row_offsets.size(); i++) {
    ASSERT_EQ(A_mirror.row_offsets[i], 1)
        << "Value of the mirror row offsets should be the same as the value of "
           "A.row_offsets during deep copy (1)";
  }
  for (index_type i = 0; i < A_mirror.nnnz(); i++) {
    ASSERT_EQ(A_mirror.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the value "
           "of A.column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "A.values during deep copy (5)";
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
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_SameSpace_MirrorContainer) {
  using container =
      Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  index_array_type Ar_rind(5, 0), Ar_cind(2, 1);
  value_array_type Ar_val(2, -1);
  container Ar("Csr", 4, 3, 2, Ar_rind, Ar_cind, Ar_val);

  // Might perform shallow copy if already on host
  auto A_mirror = Morpheus::create_mirror_container(A);

  // if on host x_mirror should be a shallow copy of x
  Morpheus::copy(A, A_mirror);
  Morpheus::copy(Ar, A);

  check_shapes(A, A_mirror, Morpheus::CsrTag{});

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space>) {
    for (index_type i = 0; i < A_mirror.row_offsets.size(); i++) {
      ASSERT_EQ(A_mirror.row_offsets[i], Ar.row_offsets[i])
          << "Value of the mirror row offsets should be the same as the value  "
             "of Ar.row_offsets (2) due to shallow copy";
    }
    for (index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.column_indices[i], Ar.column_indices[i])
          << "Value of the mirror column indices should be the same as the "
             "value of Ar.column_indices (1) due to shallow copy";
      ASSERT_EQ(A_mirror.values[i], Ar.values[i])
          << "Value of the mirror values should be the same as the value of "
             "Ar.values (3) due to shallow copy";
    }
  } else {
    for (index_type i = 0; i < A_mirror.row_offsets.size(); i++) {
      ASSERT_EQ(A_mirror.row_offsets[i], 1)
          << "Value of the mirror row offsets should be the same as the value "
             "of A.row_offsets during deep copy (1)";
    }
    for (index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.column_indices[i], 2)
          << "Value of the mirror column indices should be the same as the  "
             "value of A.column_indices during deep copy (2)";
      ASSERT_EQ(A_mirror.values[i], 5)
          << "Value of the mirror values should be the same as the value of "
             "A.values during deep copy (5)";
    }
  }
}

#if defined(MORPHEUS_ENABLE_CUDA)

// Creates a mirror on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_DeviceHost) {
  using container = Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft,
                                        typename Kokkos::Cuda>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // DtoH

  check_shapes(A, A_mirror, Morpheus::CsrTag{});

  for (index_type i = 0; i < A_mirror.row_offsets.size(); i++) {
    ASSERT_EQ(A_mirror.row_offsets[i], 1)
        << "Value of the mirror row offsets should be the same as the value "
           "of the device row_offsets during deep copy (1)";
  }
  for (index_type i = 0; i < A_mirror.nnnz(); i++) {
    ASSERT_EQ(A_mirror.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of the device column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "the device values during deep copy (5)";
  }
}

// Creates a mirror container on host from device
// Issues a copy between device and host, which is always deep
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_DeviceHost_MirrorCotnainer) {
  using container = Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft,
                                        typename Kokkos::Cuda>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  auto A_mirror = Morpheus::create_mirror_container(A);

  Morpheus::copy(A, A_mirror);  // DtoH

  check_shapes(A, A_mirror, Morpheus::CsrTag{});

  for (index_type i = 0; i < A_mirror.row_offsets.size(); i++) {
    ASSERT_EQ(A_mirror.row_offsets[i], 1)
        << "Value of the mirror row offsets should be the same as the value "
           "of the device row_offsets during deep copy (1)";
  }
  for (index_type i = 0; i < A_mirror.nnnz(); i++) {
    ASSERT_EQ(A_mirror.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of the device column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "the device values during deep copy (5)";
  }
}

// Creates a mirror on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_HostDevice) {
  using container = Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft,
                                        typename Kokkos::HostSpace>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  auto A_mirror_dev  = Morpheus::create_mirror<Kokkos::Cuda>(A);
  auto A_mirror_host = Morpheus::create_mirror(A_mirror_dev);

  Morpheus::copy(A, A_mirror_dev);              // HtoD
  Morpheus::copy(A_mirror_dev, A_mirror_host);  // DtoH

  check_shapes(A, A_mirror_host, Morpheus::CsrTag{});
  check_shapes(A, A_mirror_dev, Morpheus::CsrTag{});

  // Change the value to main container to check if we did shallow copy
  A.row_offsets.assign(A.row_offsets.size(), 0);
  A.column_indices.assign(A.column_indices.size(), 1);
  A.values.assign(A.values.size(), -1);

  for (index_type i = 0; i < A_mirror_host.row_offsets.size(); i++) {
    ASSERT_EQ(A_mirror_host.row_offsets[i], 1)
        << "Value of the mirror row offsets should be the same as the value "
           "of A.row_offsets during deep copy (1)";
  }
  for (index_type i = 0; i < A_mirror_host.nnnz(); i++) {
    ASSERT_EQ(A_mirror_host.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of A.column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror_host.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "A.values during deep copy (5)";
  }
}

// Creates a mirror on device from host
// Issues a copy between host to device and back (both should always be deep)
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_HostDevice_MirrorContainer) {
  using container = Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft,
                                        typename Kokkos::HostSpace>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  auto A_mirror_dev  = Morpheus::create_mirror_container<Kokkos::Cuda>(A);
  auto A_mirror_host = Morpheus::create_mirror_container(A_mirror_dev);

  Morpheus::copy(A, A_mirror_dev);              // HtoD
  Morpheus::copy(A_mirror_dev, A_mirror_host);  // DtoH

  check_shapes(A, A_mirror_host, Morpheus::CsrTag{});
  check_shapes(A, A_mirror_dev, Morpheus::CsrTag{});

  // Change the value to main container to check if we did shallow copy
  A.row_offsets.assign(A.row_offsets.size(), 0);
  A.column_indices.assign(A.column_indices.size(), 1);
  A.values.assign(A.values.size(), -1);

  for (index_type i = 0; i < A_mirror_host.row_offsets.size(); i++) {
    ASSERT_EQ(A_mirror_host.row_offsets[i], 1)
        << "Value of the mirror row offsets should be the same as the value "
           "of A.row_offsets during deep copy (1)";
  }
  for (index_type i = 0; i < A_mirror_host.nnnz(); i++) {
    ASSERT_EQ(A_mirror_host.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of A.column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror_host.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "A.values during deep copy (5)";
  }
}

// Creates two mirror containers on device from host
// Issues a copy between host to device and then a copy between
// the two device mirrors. Then sends the result back to host
// to be compared which should match the initial state of Ar.
TEST(TESTSUITE_NAME, DeepCopy_CsrMatrix_DeviecDevice_MirrorContainer) {
  using container = Morpheus::CsrMatrix<float, long long, Kokkos::LayoutLeft,
                                        typename Kokkos::HostSpace>;
  using index_array_type = typename container::index_array_type;
  using value_array_type = typename container::value_array_type;
  using index_type       = typename container::index_type;

  index_array_type A_roff(5, 1), A_cind(2, 2);
  value_array_type A_val(2, 5);
  container A("Csr", 4, 3, 2, A_roff, A_cind, A_val);

  index_array_type Ar_rind(5, 0), Ar_cind(2, 1);
  value_array_type Ar_val(2, -1);
  container Ar("Csr", 4, 3, 2, Ar_rind, Ar_cind, Ar_val);

  auto A_mirror_dev1 = Morpheus::create_mirror_container<Kokkos::Cuda>(A);
  auto A_mirror_dev2 = Morpheus::create_mirror_container<Kokkos::Cuda>(Ar);
  auto Ares          = Morpheus::create_mirror(A_mirror_dev1);

  Morpheus::copy(A, A_mirror_dev1);              // HtoD
  Morpheus::copy(Ar, A_mirror_dev2);             // HtoD
  Morpheus::copy(A_mirror_dev2, A_mirror_dev1);  // DtoD
  Morpheus::copy(A_mirror_dev1, Ares);           // DtoH

  check_shapes(A, Ares, Morpheus::CsrTag{});
  // Change the value to main container to check if we did shallow copy
  A.row_offsets.assign(A.row_offsets.size(), 3);
  A.column_indices.assign(A.column_indices.size(), 8);
  A.values.assign(A.values.size(), 9);
  Ar.row_offsets.assign(A.row_offsets.size(), 3);
  Ar.column_indices.assign(A.column_indices.size(), 8);
  Ar.values.assign(A.values.size(), 9);

  for (index_type i = 0; i < Ares.row_offsets.size(); i++) {
    ASSERT_EQ(Ares.row_offsets[i], 0)
        << "Value of the Ares.row_offsets should be the same as the initial "
           "value of Ar.row_offsets during deep copy (0)";
  }
  for (index_type i = 0; i < Ares.nnnz(); i++) {
    ASSERT_EQ(Ares.column_indices[i], 1)
        << "Value of the Ares.column_indices should be the same as the "
           "initial value of Ar.column_indices during deep copy (1)";
    ASSERT_EQ(Ares.values[i], -1)
        << "Value of the Ares.values should be the "
           "same as the initial value of Ar.values during deep copy (-1)";
  }
}

#endif  // MORPHEUS_ENABLE_CUDA

}  // namespace Test