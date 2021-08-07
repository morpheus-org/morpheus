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
TEST(TESTSUITE_NAME, Mirror_CooMatrix_HostMirror) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  typename container::index_array_type rind(2, 1), cind(2, 2);
  typename container::value_array_type val(2, 5);
  container A("Coo", 3, 2, 2, rind, cind, val);

  auto A_mirror = Morpheus::create_mirror(A);
  using mirror  = decltype(A_mirror);

  static_assert(
      std::is_same<typename mirror::type,
                   typename container::HostMirror>::value,
      "Mirror type should match the HostMirror type of the original container "
      "as we are creating a mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(A.nrows(), A_mirror.nrows());
    ASSERT_EQ(A.ncols(), A_mirror.ncols());
    ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
    for (typename mirror::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.row_indices[i], 0)
          << "Value of the mirror row indices should be the default "
             "(0) i.e no copy was performed";
      ASSERT_EQ(A_mirror.column_indices[i], 0)
          << "Value of the mirror column indices should be the default "
             "(0) i.e no copy was performed";
      ASSERT_EQ(A_mirror.values[i], 0)
          << "Value of the mirror values should be the default "
             "(0) i.e no copy was performed";
    }
  }
}

TEST(TESTSUITE_NAME, Mirror_CooMatrix_explicit_space) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<typename TEST_EXECSPACE::memory_space>,
      Kokkos::Cuda, Kokkos::DefaultHostExecutionSpace>;
  using dst_type =
      Morpheus::CooMatrix<typename container::value_type,
                          typename container::index_type,
                          typename container::array_layout, mirror_space>;

  typename container::index_array_type rind(2, 1), cind(2, 2);
  typename container::value_array_type val(2, 5);
  container A("Coo", 3, 2, 2, rind, cind, val);

  auto A_mirror = Morpheus::create_mirror<mirror_space>(A);
  using mirror  = decltype(A_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");

  ASSERT_EQ(A.nrows(), A_mirror.nrows());
  ASSERT_EQ(A.ncols(), A_mirror.ncols());
  ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
  if (Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    for (typename mirror::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.row_indices[i], 0)
          << "Value of the mirror row indices should be the default "
             "(0) i.e no copy was performed";
      ASSERT_EQ(A_mirror.column_indices[i], 0)
          << "Value of the mirror column indices should be the default "
             "(0) i.e no copy was performed";
      ASSERT_EQ(A_mirror.values[i], 0)
          << "Value of the mirror values should be the default "
             "(0) i.e no copy was performed";
    }
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_CooMatrix_HostMirror) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  typename container::index_array_type rind(2, 1), cind(2, 2);
  typename container::value_array_type val(2, 5);
  container A("Coo", 3, 2, 2, rind, cind, val);
  auto A_mirror = Morpheus::create_mirror_container(A);
  using mirror  = decltype(A_mirror);

  static_assert(std::is_same<typename mirror::type,
                             typename container::HostMirror::type>::value,
                "Source and mirror types must be the same as we are creating a "
                "mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(A.nrows(), A_mirror.nrows());
    ASSERT_EQ(A.ncols(), A_mirror.ncols());
    ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
    // Change the value to main container to check if we did shallow copy
    for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A.row_indices[i], 1)
          << "Value of the Coo row indices should be (1)";
      ASSERT_EQ(A.column_indices[i], 2)
          << "Value of the Coo column indices should be (2)";
      ASSERT_EQ(A.values[i], 5) << "Value of the Coo values should be (5)";
    }

    for (typename container::index_type i = 0; i < A_mirror.nnnz(); i++) {
      ASSERT_EQ(A_mirror.row_indices[i], 1)
          << "Value of the mirror row indices should be (1)";
      ASSERT_EQ(A_mirror.column_indices[i], 2)
          << "Value of the mirror column indices should be (2)";
      ASSERT_EQ(A_mirror.values[i], 5)
          << "Value of the mirror values should be (5)";
    }
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_CooMatrix_explicit_same_space) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;

  typename container::index_array_type rind(2, 1), cind(2, 2);
  typename container::value_array_type val(2, 5);
  container A("Coo", 3, 2, 2, rind, cind, val);
  auto A_mirror = Morpheus::create_mirror_container<TEST_EXECSPACE>(A);
  using mirror  = decltype(A_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename container::type>::value,
      "Source and mirror types must be the same as we are creating a "
      "mirror in the same space.");

  if (Morpheus::is_Host_Memoryspace_v<typename container::memory_space> &&
      Morpheus::is_Host_Memoryspace_v<typename mirror::memory_space>) {
    ASSERT_EQ(A.nrows(), A_mirror.nrows());
    ASSERT_EQ(A.ncols(), A_mirror.ncols());
    ASSERT_EQ(A.nnnz(), A_mirror.nnnz());
    // Change the value to main container to check if we did shallow copy
    A.row_indices[0]    = 0;
    A.column_indices[0] = 1;
    A.values[0]         = -1;
    A.row_indices[1]    = 1;
    A.column_indices[1] = 0;
    A.values[1]         = 3;

    ASSERT_EQ(A.row_indices[0], 0)
        << "Value of the Coo row indices should be (1)";
    ASSERT_EQ(A.column_indices[0], 1)
        << "Value of the Coo column indices should be (2)";
    ASSERT_EQ(A.values[0], -1) << "Value of the Coo values should be (5)";
    ASSERT_EQ(A.row_indices[1], 1)
        << "Value of the Coo row indices should be (1)";
    ASSERT_EQ(A.column_indices[1], 0)
        << "Value of the Coo column indices should be (2)";
    ASSERT_EQ(A.values[1], 3) << "Value of the Coo values should be (5)";
  }
}

TEST(TESTSUITE_NAME, MirrorContainer_CooMatrix_explicit_new_space) {
  using container =
      Morpheus::CooMatrix<float, long long, Kokkos::LayoutLeft, TEST_EXECSPACE>;
  using mirror_space = std::conditional_t<
      Morpheus::is_Host_Memoryspace_v<typename TEST_EXECSPACE::memory_space>,
      Kokkos::Cuda, Kokkos::DefaultHostExecutionSpace>;
  using dst_type =
      Morpheus::CooMatrix<typename container::value_type,
                          typename container::index_type,
                          typename container::array_layout, mirror_space>;

  typename container::index_array_type rind(2, 1), cind(2, 2);
  typename container::value_array_type val(2, 5);
  container A("Coo", 3, 2, 2, rind, cind, val);
  auto A_mirror = Morpheus::create_mirror_container<mirror_space>(A);
  using mirror  = decltype(A_mirror);

  static_assert(
      std::is_same<typename mirror::type, typename dst_type::type>::value,
      "Mirror type should be the same as the source type but in the new mirror "
      "space.");
}

}  // namespace Test