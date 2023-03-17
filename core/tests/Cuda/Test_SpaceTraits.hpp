/**
 * Test_SpaceTraits.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef TEST_CORE_CUDA_TEST_SPACETRAITS_HPP
#define TEST_CORE_CUDA_TEST_SPACETRAITS_HPP

#include <Morpheus_Core.hpp>

namespace Impl {

template <typename T>
struct with_memspace {
  using memory_space = T;
};

// A structure like this meets the requirements of a valid memory space i.e
// has a memory_space trait that is the same as it's name BUT this is not
// supported as a MemorySpace
struct TestSpace {
  using memory_space = TestSpace;
};
}  // namespace Impl

namespace Test {
#define MORPHEUS_CHECK_SPACE(REL, SPACE, TRAIT, ref_res)                   \
  {                                                                        \
    bool _res;                                                             \
    _res = Morpheus::REL<Kokkos::SPACE>::value;                            \
    EXPECT_EQ(_res, ref_res[0]);                                           \
    _res = Morpheus::REL<Morpheus::SPACE>::value;                          \
    EXPECT_EQ(_res, ref_res[1]);                                           \
    _res = Morpheus::REL<Morpheus::Custom::SPACE>::value;                  \
    EXPECT_EQ(_res, ref_res[2]);                                           \
    _res = Morpheus::REL<Morpheus::Generic::SPACE>::value;                 \
    EXPECT_EQ(_res, ref_res[3]);                                           \
    _res = Morpheus::REL<typename Kokkos::SPACE::TRAIT>::value;            \
    EXPECT_EQ(_res, ref_res[4]);                                           \
    _res = Morpheus::REL<typename Morpheus::SPACE::TRAIT>::value;          \
    EXPECT_EQ(_res, ref_res[5]);                                           \
    _res = Morpheus::REL<typename Morpheus::Custom::SPACE::TRAIT>::value;  \
    EXPECT_EQ(_res, ref_res[6]);                                           \
    _res = Morpheus::REL<typename Morpheus::Generic::SPACE::TRAIT>::value; \
    EXPECT_EQ(_res, ref_res[7]);                                           \
    /* Checking Alias */                                                   \
    _res = Morpheus::REL##_v<Kokkos::SPACE>;                               \
    EXPECT_EQ(_res, ref_res[0]);                                           \
    _res = Morpheus::REL##_v<Morpheus::SPACE>;                             \
    EXPECT_EQ(_res, ref_res[1]);                                           \
    _res = Morpheus::REL##_v<Morpheus::Custom::SPACE>;                     \
    EXPECT_EQ(_res, ref_res[2]);                                           \
    _res = Morpheus::REL##_v<Morpheus::Generic::SPACE>;                    \
    EXPECT_EQ(_res, ref_res[3]);                                           \
    _res = Morpheus::REL##_v<typename Kokkos::SPACE::TRAIT>;               \
    EXPECT_EQ(_res, ref_res[4]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::SPACE::TRAIT>;             \
    EXPECT_EQ(_res, ref_res[5]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::Custom::SPACE::TRAIT>;     \
    EXPECT_EQ(_res, ref_res[6]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::Generic::SPACE::TRAIT>;    \
    EXPECT_EQ(_res, ref_res[7]);                                           \
  }

#define MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(REL, SPACE1, SPACE2, TRAIT,        \
                                            ref_res)                           \
  {                                                                            \
    bool _res;                                                                 \
    _res = Morpheus::REL<Kokkos::SPACE1, Kokkos::SPACE2>::value;               \
    EXPECT_EQ(_res, ref_res[0]);                                               \
    _res = Morpheus::REL<Morpheus::SPACE1, Morpheus::SPACE2>::value;           \
    EXPECT_EQ(_res, ref_res[1]);                                               \
    _res = Morpheus::REL<Morpheus::Custom::SPACE1,                             \
                         Morpheus::Custom::SPACE2>::value;                     \
    EXPECT_EQ(_res, ref_res[2]);                                               \
    _res = Morpheus::REL<Morpheus::Generic::SPACE1,                            \
                         Morpheus::Generic::SPACE2>::value;                    \
    EXPECT_EQ(_res, ref_res[3]);                                               \
    _res = Morpheus::REL<typename Kokkos::SPACE1::TRAIT,                       \
                         typename Kokkos::SPACE2::TRAIT>::value;               \
    EXPECT_EQ(_res, ref_res[4]);                                               \
    _res = Morpheus::REL<typename Morpheus::SPACE1::TRAIT,                     \
                         typename Morpheus::SPACE2::TRAIT>::value;             \
    EXPECT_EQ(_res, ref_res[5]);                                               \
    _res = Morpheus::REL<typename Morpheus::Custom::SPACE1::TRAIT,             \
                         typename Morpheus::Custom::SPACE2::TRAIT>::value;     \
    EXPECT_EQ(_res, ref_res[6]);                                               \
    _res = Morpheus::REL<typename Morpheus::Generic::SPACE1::TRAIT,            \
                         typename Morpheus::Generic::SPACE2::TRAIT>::value;    \
    EXPECT_EQ(_res, ref_res[7]);                                               \
    /* Checking Alias */                                                       \
    _res = Morpheus::REL##_v<Kokkos::SPACE1, Kokkos::SPACE2>;                  \
    EXPECT_EQ(_res, ref_res[0]);                                               \
    _res = Morpheus::REL##_v<Morpheus::SPACE1, Morpheus::SPACE2>;              \
    EXPECT_EQ(_res, ref_res[1]);                                               \
    _res =                                                                     \
        Morpheus::REL##_v<Morpheus::Custom::SPACE1, Morpheus::Custom::SPACE2>; \
    EXPECT_EQ(_res, ref_res[2]);                                               \
    _res = Morpheus::REL##_v<Morpheus::Generic::SPACE1,                        \
                             Morpheus::Generic::SPACE2>;                       \
    EXPECT_EQ(_res, ref_res[3]);                                               \
    _res = Morpheus::REL##_v<typename Kokkos::SPACE1::TRAIT,                   \
                             typename Kokkos::SPACE2::TRAIT>;                  \
    EXPECT_EQ(_res, ref_res[4]);                                               \
    _res = Morpheus::REL##_v<typename Morpheus::SPACE1::TRAIT,                 \
                             typename Morpheus::SPACE2::TRAIT>;                \
    EXPECT_EQ(_res, ref_res[5]);                                               \
    _res = Morpheus::REL##_v<typename Morpheus::Custom::SPACE1::TRAIT,         \
                             typename Morpheus::Custom::SPACE2::TRAIT>;        \
    EXPECT_EQ(_res, ref_res[6]);                                               \
    _res = Morpheus::REL##_v<typename Morpheus::Generic::SPACE1::TRAIT,        \
                             typename Morpheus::Generic::SPACE2::TRAIT>;       \
    EXPECT_EQ(_res, ref_res[7]);                                               \
  }

#define MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(REL, SPACE1, SPACE2, TRAIT,       \
                                            ref_res)                          \
  {                                                                           \
    bool _res;                                                                \
    _res = Morpheus::REL<Kokkos::SPACE1, Morpheus::SPACE2>::value;            \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::REL<Kokkos::SPACE1, Morpheus::Custom::SPACE2>::value;    \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::REL<Kokkos::SPACE1, Morpheus::Generic::SPACE2>::value;   \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::REL<Morpheus::SPACE1, Morpheus::Custom::SPACE2>::value;  \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::REL<Morpheus::SPACE1, Morpheus::Generic::SPACE2>::value; \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::REL<Morpheus::Custom::SPACE1,                            \
                         Morpheus::Generic::SPACE2>::value;                   \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::REL<typename Kokkos::SPACE1::TRAIT,                      \
                         typename Morpheus::SPACE2::TRAIT>::value;            \
    EXPECT_EQ(_res, ref_res[6]);                                              \
    _res = Morpheus::REL<typename Kokkos::SPACE1::TRAIT,                      \
                         typename Morpheus::Custom::SPACE2::TRAIT>::value;    \
    EXPECT_EQ(_res, ref_res[7]);                                              \
    _res = Morpheus::REL<typename Kokkos::SPACE1::TRAIT,                      \
                         typename Morpheus::Generic::SPACE2::TRAIT>::value;   \
    EXPECT_EQ(_res, ref_res[8]);                                              \
    _res = Morpheus::REL<typename Morpheus::SPACE1::TRAIT,                    \
                         typename Morpheus::Custom::SPACE2::TRAIT>::value;    \
    EXPECT_EQ(_res, ref_res[9]);                                              \
    _res = Morpheus::REL<typename Morpheus::SPACE1::TRAIT,                    \
                         typename Morpheus::Generic::SPACE2::TRAIT>::value;   \
    EXPECT_EQ(_res, ref_res[10]);                                             \
    _res = Morpheus::REL<typename Morpheus::Custom::SPACE1::TRAIT,            \
                         typename Morpheus::Generic::SPACE2::TRAIT>::value;   \
    EXPECT_EQ(_res, ref_res[11]);                                             \
    /* Checking Alias */                                                      \
    _res = Morpheus::REL##_v<Kokkos::SPACE1, Morpheus::SPACE2>;               \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::REL##_v<Kokkos::SPACE1, Morpheus::Custom::SPACE2>;       \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::REL##_v<Kokkos::SPACE1, Morpheus::Generic::SPACE2>;      \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::REL##_v<Morpheus::SPACE1, Morpheus::Custom::SPACE2>;     \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::REL##_v<Morpheus::SPACE1, Morpheus::Generic::SPACE2>;    \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::REL##_v<Morpheus::Custom::SPACE1,                        \
                             Morpheus::Generic::SPACE2>;                      \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::REL##_v<typename Kokkos::SPACE1::TRAIT,                  \
                             typename Morpheus::SPACE2::TRAIT>;               \
    EXPECT_EQ(_res, ref_res[6]);                                              \
    _res = Morpheus::REL##_v<typename Kokkos::SPACE1::TRAIT,                  \
                             typename Morpheus::Custom::SPACE2::TRAIT>;       \
    EXPECT_EQ(_res, ref_res[7]);                                              \
    _res = Morpheus::REL##_v<typename Kokkos::SPACE1::TRAIT,                  \
                             typename Morpheus::Generic::SPACE2::TRAIT>;      \
    EXPECT_EQ(_res, ref_res[8]);                                              \
    _res = Morpheus::REL##_v<typename Morpheus::SPACE1::TRAIT,                \
                             typename Morpheus::Custom::SPACE2::TRAIT>;       \
    EXPECT_EQ(_res, ref_res[9]);                                              \
    _res = Morpheus::REL##_v<typename Morpheus::SPACE1::TRAIT,                \
                             typename Morpheus::Generic::SPACE2::TRAIT>;      \
    EXPECT_EQ(_res, ref_res[10]);                                             \
    _res = Morpheus::REL##_v<typename Morpheus::Custom::SPACE1::TRAIT,        \
                             typename Morpheus::Generic::SPACE2::TRAIT>;      \
    EXPECT_EQ(_res, ref_res[11]);                                             \
  }

/**
 * @brief The \p is_memory_space checks if the passed type is a valid memory
 * space. For the check to be valid, the type must be one of the supported
 * memory spaces. Note that Morpheus spaces are not the same as Kokkos
 * spaces - they have a memory and execution space instead of being one.
 *
 */
TEST(SpaceTraitsTest, IsMemorySpaceCudaSpace) {
  {
    bool ref_results[8] = {1, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(is_memory_space, CudaSpace, memory_space, ref_results);
  }

  // Check Execution Spaces
  {
    bool ref_results[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(is_memory_space, DefaultExecutionSpace, memory_space,
                         ref_results);
  }

  {
    bool ref_results[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(is_memory_space, TEST_SPACE, memory_space,
                         ref_results);
  }
}

/**
 * @brief The \p has_memory_space checks if the passed type has a valid
 memory
 * space. For the check to be valid, the type must be a valid memory space
 and
 * have a \p memory_space trait.
 *
 */
TEST(SpaceTraitsTest, HasMemorySpaceCudaSpace) {
  {
    bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(has_memory_space, CudaSpace, memory_space,
                         ref_results);
  }

  {
    bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(has_memory_space, TEST_SPACE, memory_space,
                         ref_results);
  }
}

/**
 * @brief The \p is_execution_space checks if the passed type is a valid
 * executions space. For the check to be valid, the type must be one of the
 * supported execution spaces. Note that Morpheus spaces are not the same as
 * Kokkos spaces - they have a memory and execution space instead of being one.
 *
 */
TEST(SpaceTraitsTest, IsExecutionSpaceCuda) {
  bool ref_results[8] = {1, 0, 0, 0, 1, 1, 1, 1};
  MORPHEUS_CHECK_SPACE(is_execution_space, Cuda, execution_space, ref_results);
}

/**
 * @brief The \p has_execution_space checks if the passed type has a valid
 * executions space. For the check to be valid, the type must have one of the
 * supported execution spaces as its execution_space trait. Note that Morpheus
 * spaces are not the same as Kokkos spaces - they have a memory and execution
 * space instead of being one.
 *
 */
TEST(SpaceTraitsTest, HasExecutionSpaceCuda) {
  bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  MORPHEUS_CHECK_SPACE(has_execution_space, Cuda, execution_space, ref_results);
}

/**
 * @brief The \p is_same_memory_space checks if the two types passed are the
 * same memory space. For the check to be valid, both types must be a valid
 * memory space and be the same.
 *
 */
TEST(SpaceTraitsTest, IsSameMemorySpaceCudaSpace) {
  {
    bool ref_results[8] = {1, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(is_same_memory_space, CudaSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(is_same_memory_space, CudaSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(is_same_memory_space, HostSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(is_same_memory_space, HostSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  // Execution Space
  {
    bool ref_results[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(is_same_memory_space, TEST_SPACE,
                                        TEST_SPACE, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(is_same_memory_space, TEST_SPACE,
                                        TEST_SPACE, memory_space, ref_results);
  }
}

/**
 * @brief The \p has_same_memory_space checks if the two types passed have
 the
 * same memory space. For the check to be valid, both types must have a
 * \p memory_space trait and the \p is_same_memory_space must be satisfied.
 *
 */
TEST(SpaceTraitsTest, HasSameMemorySpaceCudaSpace) {
  {
    bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(has_same_memory_space, CudaSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(has_same_memory_space, CudaSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(has_same_memory_space, HostSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(has_same_memory_space, HostSpace,
                                        CudaSpace, memory_space, ref_results);
  }

  // Execution Space
  {
    bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_SAME_NAMESPACE(has_same_memory_space, TEST_SPACE,
                                        TEST_SPACE, memory_space, ref_results);
  }

  {
    bool ref_results[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE_DIFF_NAMESPACE(has_same_memory_space, TEST_SPACE,
                                        TEST_SPACE, memory_space, ref_results);
  }
}

/**
 * @brief The \p is_host_memory_space checks if the passed type is a valid
 * Host memory space. For the check to be valid, the type must be one of the
 * supported Host memory spaces.
 *
 */
TEST(SpaceTraitsTest, IsHostMemorySpaceCudaSpace) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(is_host_memory_space, CudaSpace, memory_space,
                       ref_results);
}

/**
 * @brief The \p has_host_memory_space checks if the passed type has a valid
 * Host memory space. For the check to be valid, the type must hold one of the
 * supported Host memory spaces.
 *
 */
TEST(SpaceTraitsTest, HasHostMemorySpaceCudaSpace) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(has_host_memory_space, CudaSpace, memory_space,
                       ref_results);
}

/**
 * @brief The \p is_host_execution_space checks if the passed type is a valid
 * Host executions space. For the check to be valid, the type must be one of
 * the supported Host execution spaces.
 *
 */
TEST(SpaceTraitsTest, IsHostExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(is_host_execution_space, Cuda, execution_space,
                       ref_results);
}

/**
 * @brief The \p has_host_execution_space checks if the passed type has a valid
 * Host executions space. For the check to be valid, the type must be one of
 * the supported Host execution spaces.
 *
 */
TEST(SpaceTraitsTest, HasHostExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(has_host_execution_space, Cuda, execution_space,
                       ref_results);
}

#if defined(MORPHEUS_ENABLE_SERIAL)
/**
 * @brief The \p is_serial_execution_space checks if the passed type is a
 * valid Serial executions space. For the check to be valid, the type must be
 * a Serial execution space.
 *
 */
TEST(SpaceTraitsTest, IsSerialExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(is_serial_execution_space, Cuda, execution_space,
                       ref_results);
}

/**
 * @brief The \p has_serial_execution_space checks if the passed type has a
 * valid Serial executions space. For the check to be valid, the type must hold
 * a Serial execution space.
 *
 */
TEST(SpaceTraitsTest, HasSerialExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(has_serial_execution_space, Cuda, execution_space,
                       ref_results);
}
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
/**
 * @brief The \p is_openmp_execution_space checks if the passed type is a
 * valid OpenMP executions space. For the check to be valid, the type must be
 * a OpenMP execution space.
 *
 */
TEST(SpaceTraitsTest, IsOpenMPExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(is_openmp_execution_space, Cuda, execution_space,
                       ref_results);
}

/**
 * @brief The \p has_openmp_execution_space checks if the passed type has a
 * valid OpenMP execution space. For the check to be valid, the type must hold
 * a OpenMP execution space.
 *
 */
TEST(SpaceTraitsTest, HasOpenMPExecutionSpaceCuda) {
  bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MORPHEUS_CHECK_SPACE(has_openmp_execution_space, Cuda, execution_space,
                       ref_results);
}
#endif  // MORPHEUS_ENABLE_OPENMP

/**
 * @brief The \p is_cuda_execution_space checks if the passed type is a
 * valid Cuda execution space. For the check to be valid, the type must be
 * a Cuda execution space.
 *
 */
TEST(SpaceTraitsTest, IsCudaExecutionSpaceCuda) {
  {
    bool result = Morpheus::is_cuda_execution_space<int>::value;
    EXPECT_EQ(result, 0);

    result = Morpheus::is_cuda_execution_space_v<int>;
    EXPECT_EQ(result, 0);
  }

  {
    struct A {};
    bool result = Morpheus::is_cuda_execution_space<A>::value;
    EXPECT_EQ(result, 0);

    result = Morpheus::is_cuda_execution_space_v<A>;
    EXPECT_EQ(result, 0);
  }

#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(is_cuda_execution_space, Serial, execution_space,
                         ref_results);
  }
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(is_cuda_execution_space, OpenMP, execution_space,
                         ref_results);
  }
#endif

  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(is_cuda_execution_space, DefaultHostExecutionSpace,
                         execution_space, ref_results);
  }

  {
    bool ref_results[8] = {1, 0, 0, 0, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(is_cuda_execution_space, Cuda, execution_space,
                         ref_results);
  }
}

/**
 * @brief The \p has_cuda_execution_space checks if the passed type has a
 * valid Cuda execution space. For the check to be valid, the type must hold
 * a Cuda execution space.
 *
 */
TEST(SpaceTraitsTest, HasCudaExecutionSpaceCuda) {
  {
    bool result = Morpheus::has_cuda_execution_space<int>::value;
    EXPECT_EQ(result, 0);

    result = Morpheus::has_cuda_execution_space_v<int>;
    EXPECT_EQ(result, 0);
  }

  {
    struct A {};
    bool result = Morpheus::has_cuda_execution_space<A>::value;
    EXPECT_EQ(result, 0);

    result = Morpheus::has_cuda_execution_space_v<A>;
    EXPECT_EQ(result, 0);
  }

#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(has_cuda_execution_space, Serial, execution_space,
                         ref_results);
  }
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(has_cuda_execution_space, OpenMP, execution_space,
                         ref_results);
  }
#endif

  {
    bool ref_results[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_SPACE(has_cuda_execution_space, DefaultHostExecutionSpace,
                         execution_space, ref_results);
  }

  {
    bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_SPACE(has_cuda_execution_space, Cuda, execution_space,
                         ref_results);
  }
}

#define MORPHEUS_CHECK_ACCESS_SINGLE_SPACE_NAMESPACE(                          \
    REL, NAMESPACE, SPACE, MEMSPACE, _ref_res, res_offset)                     \
  {                                                                            \
    bool _res =                                                                \
        Morpheus::has_access<NAMESPACE::SPACE, Kokkos::MEMSPACE>::value;       \
    EXPECT_EQ(_res, _ref_res[res_offset + 0]);                                 \
    _res = Morpheus::has_access<NAMESPACE::SPACE, Morpheus::MEMSPACE>::value;  \
    EXPECT_EQ(_res, _ref_res[res_offset + 1]);                                 \
    _res = Morpheus::has_access<NAMESPACE::SPACE,                              \
                                Morpheus::Custom::MEMSPACE>::value;            \
    EXPECT_EQ(_res, _ref_res[res_offset + 2]);                                 \
    _res = Morpheus::has_access<NAMESPACE::SPACE,                              \
                                Morpheus::Generic::MEMSPACE>::value;           \
    EXPECT_EQ(_res, _ref_res[res_offset + 3]);                                 \
    /*Check Aliasing*/                                                         \
    _res = Morpheus::has_access_v<NAMESPACE::SPACE, Kokkos::MEMSPACE>;         \
    EXPECT_EQ(_res, _ref_res[res_offset + 0]);                                 \
    _res = Morpheus::has_access_v<NAMESPACE::SPACE, Morpheus::MEMSPACE>;       \
    EXPECT_EQ(_res, _ref_res[res_offset + 1]);                                 \
    _res =                                                                     \
        Morpheus::has_access_v<NAMESPACE::SPACE, Morpheus::Custom::MEMSPACE>;  \
    EXPECT_EQ(_res, _ref_res[res_offset + 2]);                                 \
    _res =                                                                     \
        Morpheus::has_access_v<NAMESPACE::SPACE, Morpheus::Generic::MEMSPACE>; \
    EXPECT_EQ(_res, _ref_res[res_offset + 3]);                                 \
  }

#define MORPHEUS_CHECK_ACCESS_SINGLE_SPACE(REL, SPACE, MEMSPACE, ref_res)      \
  {                                                                            \
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE_NAMESPACE(REL, Kokkos, SPACE, MEMSPACE, \
                                                 ref_res, 0);                  \
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE_NAMESPACE(REL, Morpheus, SPACE,         \
                                                 MEMSPACE, ref_res, 4);        \
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE_NAMESPACE(REL, Morpheus::Custom, SPACE, \
                                                 MEMSPACE, ref_res, 8);        \
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE_NAMESPACE(                              \
        REL, Morpheus::Generic, SPACE, MEMSPACE, ref_res, 12);                 \
  }

TEST(SpaceTraitsTest, HasAccessSingleCudaSpace) {
#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    bool ref_results[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE("", Serial, CudaSpace, ref_results);
  }
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    bool ref_results[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE("", OpenMP, CudaSpace, ref_results);
  }
#endif  // MORPHEUS_ENABLE_OPENMP

  {
    bool ref_results[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE("", Cuda, HostSpace, ref_results);
  }

  {
    bool ref_results[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MORPHEUS_CHECK_ACCESS_SINGLE_SPACE("", Cuda, CudaSpace, ref_results);
  }
}

#define MORPHEUS_CHECK_ACCESS_MULTI_SPACE_NAMESPACE(                         \
    REL, NAMESPACE, SPACE, MEMSPACE, _ref_res, res_offset)                   \
  {                                                                          \
    bool _res = Morpheus::has_access<NAMESPACE::SPACE, NAMESPACE::HostSpace, \
                                     NAMESPACE::HostSpace>::value;           \
    EXPECT_EQ(_res, _ref_res[res_offset + 0]);                               \
    _res = Morpheus::has_access<NAMESPACE::SPACE, NAMESPACE::HostSpace,      \
                                NAMESPACE::MEMSPACE>::value;                 \
    EXPECT_EQ(_res, _ref_res[res_offset + 1]);                               \
    _res = Morpheus::has_access<NAMESPACE::SPACE, NAMESPACE::MEMSPACE,       \
                                NAMESPACE::MEMSPACE>::value;                 \
    EXPECT_EQ(_res, _ref_res[res_offset + 2]);                               \
    /*Check Aliasing*/                                                       \
    _res = Morpheus::has_access_v<NAMESPACE::SPACE, NAMESPACE::HostSpace,    \
                                  NAMESPACE::HostSpace>;                     \
    EXPECT_EQ(_res, _ref_res[res_offset + 0]);                               \
    _res = Morpheus::has_access_v<NAMESPACE::SPACE, NAMESPACE::HostSpace,    \
                                  NAMESPACE::MEMSPACE>;                      \
    EXPECT_EQ(_res, _ref_res[res_offset + 1]);                               \
    _res = Morpheus::has_access_v<NAMESPACE::SPACE, NAMESPACE::MEMSPACE,     \
                                  NAMESPACE::MEMSPACE>;                      \
    EXPECT_EQ(_res, _ref_res[res_offset + 2]);                               \
  }

#define MORPHEUS_CHECK_ACCESS_MULTI_SPACE(REL, SPACE, MEMSPACE, ref_res)       \
  {                                                                            \
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE_NAMESPACE(REL, Kokkos, SPACE, MEMSPACE,  \
                                                ref_res, 0);                   \
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE_NAMESPACE(REL, Morpheus, SPACE,          \
                                                MEMSPACE, ref_res, 3);         \
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE_NAMESPACE(REL, Morpheus::Custom, SPACE,  \
                                                MEMSPACE, ref_res, 6);         \
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE_NAMESPACE(REL, Morpheus::Generic, SPACE, \
                                                MEMSPACE, ref_res, 9);         \
  }

/**
 * @brief The \p has_access checks if first type (ExecutionSpace) has access
 * to the arbitrary number of types passed. For the check to be valid, the
 * first type must be a valid execution space, the rest must have a valid
 * memory space and the execution space must be able to access them.
 *
 */
TEST(SpaceTraitsTest, HasAccessMultiCudaSpace) {
#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    bool ref_results[12] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE("", Serial, CudaSpace, ref_results);
  }
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    bool ref_results[12] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE("", OpenMP, CudaSpace, ref_results);
  }
#endif  // MORPHEUS_ENABLE_OPENMP

  {
    bool ref_results[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE("", Cuda, HostSpace, ref_results);
  }

  {
    bool ref_results[12] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    MORPHEUS_CHECK_ACCESS_MULTI_SPACE("", Cuda, CudaSpace, ref_results);
  }
}

#define MORPHEUS_CHECK_BACKEND(REL, SPACE, TRAIT, ref_res)                 \
  {                                                                        \
    bool _res;                                                             \
    _res = Morpheus::REL<Morpheus::SPACE>::value;                          \
    EXPECT_EQ(_res, ref_res[0]);                                           \
    _res = Morpheus::REL<Morpheus::Custom::SPACE>::value;                  \
    EXPECT_EQ(_res, ref_res[1]);                                           \
    _res = Morpheus::REL<Morpheus::Generic::SPACE>::value;                 \
    EXPECT_EQ(_res, ref_res[2]);                                           \
    _res = Morpheus::REL<typename Morpheus::SPACE::TRAIT>::value;          \
    EXPECT_EQ(_res, ref_res[3]);                                           \
    _res = Morpheus::REL<typename Morpheus::Custom::SPACE::TRAIT>::value;  \
    EXPECT_EQ(_res, ref_res[4]);                                           \
    _res = Morpheus::REL<typename Morpheus::Generic::SPACE::TRAIT>::value; \
    EXPECT_EQ(_res, ref_res[5]);                                           \
    /* Checking Alias */                                                   \
    _res = Morpheus::REL##_v<Morpheus::SPACE>;                             \
    EXPECT_EQ(_res, ref_res[0]);                                           \
    _res = Morpheus::REL##_v<Morpheus::Custom::SPACE>;                     \
    EXPECT_EQ(_res, ref_res[1]);                                           \
    _res = Morpheus::REL##_v<Morpheus::Generic::SPACE>;                    \
    EXPECT_EQ(_res, ref_res[2]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::SPACE::TRAIT>;             \
    EXPECT_EQ(_res, ref_res[3]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::Custom::SPACE::TRAIT>;     \
    EXPECT_EQ(_res, ref_res[4]);                                           \
    _res = Morpheus::REL##_v<typename Morpheus::Generic::SPACE::TRAIT>;    \
    EXPECT_EQ(_res, ref_res[5]);                                           \
  }

TEST(SpaceTraitsTest, IsSpaceCuda) {
  bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  MORPHEUS_CHECK_SPACE(is_space, Cuda, execution_space, ref_results);
  MORPHEUS_CHECK_SPACE(is_space, Cuda, memory_space, ref_results);
  MORPHEUS_CHECK_SPACE(is_space, Cuda, device_type, ref_results);

  bool backend_results[6] = {1, 1, 1, 1, 1, 1};
  MORPHEUS_CHECK_BACKEND(is_space, Cuda, backend, backend_results);
}

TEST(SpaceTraitsTest, IsSpaceCudaSpace) {
  bool ref_results[8] = {1, 1, 1, 1, 1, 1, 1, 1};

  MORPHEUS_CHECK_SPACE(is_space, CudaSpace, execution_space, ref_results);
  MORPHEUS_CHECK_SPACE(is_space, CudaSpace, memory_space, ref_results);
  MORPHEUS_CHECK_SPACE(is_space, CudaSpace, device_type, ref_results);

  bool backend_results[6] = {1, 1, 1, 1, 1, 1};
  MORPHEUS_CHECK_BACKEND(is_space, CudaSpace, backend, backend_results);
}

}  // namespace Test

#endif  // TEST_CORE_CUDA_TEST_SPACETRAITS_HPP
