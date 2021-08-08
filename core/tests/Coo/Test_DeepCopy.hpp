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

TEST(TESTSUITE_NAME, DeepCopy_CooMatrix_SameSpace_Mirror) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  typename container::index_array_type A_rind(2, 1), A_cind(2, 2);
  typename container::value_array_type A_val(2, 5);
  container A("Coo", 3, 2, 2, A_rind, A_cind, A_val);

  typename container::index_array_type Ar_rind(2, 2), Ar_cind(2, 1);
  typename container::value_array_type Ar_val(2, 3);
  container Ar("Coo", 3, 2, 2, Ar_rind, Ar_cind, Ar_val);

  // Always allocates new memory space
  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // Space-Host
  Morpheus::copy(Ar, A);        // Should be shallow copy

  ASSERT_EQ(A.nrows(), A_mirror.nrows());
  ASSERT_EQ(A.ncols(), A_mirror.ncols());
  ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.row_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.column_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.values.size(), A_mirror.nnnz());
  for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
    ASSERT_EQ(A_mirror.row_indices[i], 1)
        << "Value of the mirror row indices should be the same as the value of "
           "A.row_indices during deep copy (1)";
    ASSERT_EQ(A_mirror.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the value "
           "of A.column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "A.values during deep copy (5)";
  }
}

TEST(TESTSUITE_NAME, DeepCopy_CooMatrix_SameSpace_MirrorContainer) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  typename container::index_array_type A_rind(2, 1), A_cind(2, 2);
  typename container::value_array_type A_val(2, 5);
  container A("Coo", 3, 2, 2, A_rind, A_cind, A_val);

  typename container::index_array_type Ar_rind(2, 2), Ar_cind(2, 1);
  typename container::value_array_type Ar_val(2, 3);
  container Ar("Coo", 3, 2, 2, Ar_rind, Ar_cind, Ar_val);

  // Might perform shallow copy if already on host
  auto A_mirror = Morpheus::create_mirror_container(A);
  using mirror  = decltype(A_mirror);

  // if on host x_mirror should be a shallow copy of x
  Morpheus::copy(A, A_mirror);
  Morpheus::copy(Ar, A);

  ASSERT_EQ(A.nrows(), A_mirror.nrows());
  ASSERT_EQ(A.ncols(), A_mirror.ncols());
  ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.row_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.column_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.values.size(), A_mirror.nnnz());

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space>) {
    for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.row_indices[i], Ar.row_indices[i])
          << "Value of the mirror row indices should be the same as the value  "
             "of Ar.row_indices (2) due to shallow copy";
      ASSERT_EQ(A_mirror.column_indices[i], Ar.column_indices[i])
          << "Value of the mirror column indices should be the same as the "
             "value of Ar.column_indices (1) due to shallow copy";
      ASSERT_EQ(A_mirror.values[i], A_mirror.values[i])
          << "Value of the mirror values should be the same as the value of "
             "Ar.values (3) due to shallow copy";
    }
  } else {
    for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.row_indices[i], 1)
          << "Value of the mirror row indices should be the same as the value "
             "of A.row_indices during deep copy (1)";
      ASSERT_EQ(A_mirror.column_indices[i], 2)
          << "Value of the mirror column indices should be the same as the  "
             "value of A.column_indices during deep copy (2)";
      ASSERT_EQ(A_mirror.values[i], 5)
          << "Value of the mirror values should be the same as the value of "
             "A.values during deep copy (5)";
    }
  }
}

TEST(TESTSUITE_NAME, DeepCopy_CooMatrix_DeviceHost) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, Kokkos::Cuda>;

  typename container::index_array_type A_rind(2, 1), A_cind(2, 2);
  typename container::value_array_type A_val(2, 5);
  container A("Coo", 3, 2, 2, A_rind, A_cind, A_val);
  auto A_mirror = Morpheus::create_mirror(A);

  Morpheus::copy(A, A_mirror);  // DtoH

  ASSERT_EQ(A.nrows(), A_mirror.nrows());
  ASSERT_EQ(A.ncols(), A_mirror.ncols());
  ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.row_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.column_indices.size(), A_mirror.nnnz());
  ASSERT_EQ(A_mirror.values.size(), A_mirror.nnnz());

  for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
    ASSERT_EQ(A_mirror.row_indices[i], 1)
        << "Value of the mirror row indices should be the same as the value "
           "of the device row_indices during deep copy (1)";
    ASSERT_EQ(A_mirror.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of the device column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "the device values during deep copy (5)";
  }
}

TEST(TESTSUITE_NAME, DeepCopy_CooMatrix_HostDevice) {
  using container = Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft,
                                        Kokkos::HostSpace>;

  typename container::index_array_type A_rind(2, 1), A_cind(2, 2);
  typename container::value_array_type A_val(2, 5);
  container A("Coo", 3, 2, 2, A_rind, A_cind, A_val);

  auto A_mirror_dev  = Morpheus::create_mirror<Kokkos::Cuda>(A);
  auto A_mirror_host = Morpheus::create_mirror(A_mirror_dev);

  Morpheus::copy(A, A_mirror_dev);              // HtoD
  Morpheus::copy(A_mirror_dev, A_mirror_host);  // DtoH

  ASSERT_EQ(A.nrows(), A_mirror_host.nrows());
  ASSERT_EQ(A.ncols(), A_mirror_host.ncols());
  ASSERT_EQ(A.nnnz(), A_mirror_host.nnnz());
  ASSERT_EQ(A_mirror_host.row_indices.size(), A_mirror_host.nnnz());
  ASSERT_EQ(A_mirror_host.column_indices.size(), A_mirror_host.nnnz());
  ASSERT_EQ(A_mirror_host.values.size(), A_mirror_host.nnnz());

  for (typename container::index_type i = 0; i < A_mirror_host.nnnz(); i++) {
    A.row_indices[i]    = 2;
    A.column_indices[i] = 1;
    A.row_indices[i]    = -4;
    ASSERT_EQ(A_mirror_host.row_indices[i], 1)
        << "Value of the mirror row indices should be the same as the value "
           "of A.row_indices during deep copy (1)";
    ASSERT_EQ(A_mirror_host.column_indices[i], 2)
        << "Value of the mirror column indices should be the same as the  "
           "value of A.column_indices during deep copy (2)";
    ASSERT_EQ(A_mirror_host.values[i], 5)
        << "Value of the mirror values should be the same as the value of "
           "A.values during deep copy (5)";
  }
}

}  // namespace Test