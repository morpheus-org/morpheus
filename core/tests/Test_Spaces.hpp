/**
 * Test_Spaces.hpp
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

#ifndef TEST_CORE_TEST_SPACES_HPP
#define TEST_CORE_TEST_SPACES_HPP

#include <Morpheus_Core.hpp>

namespace Test {
#define MORPHEUS_CHECK_GENERIC_BACKEND(SPACE, ref_res)                        \
  {                                                                           \
    bool _res;                                                                \
    _res = Morpheus::is_generic_backend<Kokkos::SPACE>::value;                \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::SPACE>::value;              \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::Custom::SPACE>::value;      \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::is_generic_backend<Morpheus::Generic::SPACE>::value;     \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::SPACE::backend>::value;                            \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::Custom::SPACE::backend>::value;                    \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::is_generic_backend<                                      \
        typename Morpheus::Generic::SPACE::backend>::value;                   \
    EXPECT_EQ(_res, ref_res[6]);                                              \
    /* Checking Alias */                                                      \
    _res = Morpheus::is_generic_backend_v<Kokkos::SPACE>;                     \
    EXPECT_EQ(_res, ref_res[0]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::SPACE>;                   \
    EXPECT_EQ(_res, ref_res[1]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::Custom::SPACE>;           \
    EXPECT_EQ(_res, ref_res[2]);                                              \
    _res = Morpheus::is_generic_backend_v<Morpheus::Generic::SPACE>;          \
    EXPECT_EQ(_res, ref_res[3]);                                              \
    _res = Morpheus::is_generic_backend_v<typename Morpheus::SPACE::backend>; \
    EXPECT_EQ(_res, ref_res[4]);                                              \
    _res = Morpheus::is_generic_backend_v<                                    \
        typename Morpheus::Custom::SPACE::backend>;                           \
    EXPECT_EQ(_res, ref_res[5]);                                              \
    _res = Morpheus::is_generic_backend_v<                                    \
        typename Morpheus::Generic::SPACE::backend>;                          \
    EXPECT_EQ(_res, ref_res[6]);                                              \
  }

/**
 * @brief The \p is_generic_space checks if the passed type is a valid Generic
 * backend. For the check to be valid, the type should be a \p GenericBackend
 * container.
 *
 */
TEST(SpacesTest, Device) {
  // Check custom backend
  {
    using exe  = Kokkos::DefaultHostExecutionSpace;
    using mem  = Kokkos::HostSpace;
    using back = Morpheus::DefaultHostExecutionSpace;
    using dev  = Morpheus::Device<exe, mem, back>;

    bool res;

    res = std::is_same<typename dev::backend, back>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::execution_space, exe>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::memory_space, mem>::value;
    EXPECT_EQ(res, 1);
  }
  // Check generic backend
  {
    using exe  = Kokkos::DefaultHostExecutionSpace;
    using mem  = Kokkos::HostSpace;
    using back = Morpheus::Generic::DefaultHostExecutionSpace;
    using dev  = Morpheus::Device<exe, mem, back>;

    bool res;

    res = std::is_same<typename dev::backend, back>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::execution_space, exe>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::memory_space, mem>::value;
    EXPECT_EQ(res, 1);
  }
  // Check if correct execution & memory space is used
  {
    using exe  = Morpheus::DefaultHostExecutionSpace;
    using mem  = Morpheus::HostSpace;
    using back = Morpheus::DefaultHostExecutionSpace;
    using dev  = Morpheus::Device<exe, mem, back>;

    bool res;

    res = std::is_same<typename dev::backend, back>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::execution_space, exe>::value;
    EXPECT_EQ(res, 0);
    res = std::is_same<typename dev::memory_space, mem>::value;
    EXPECT_EQ(res, 0);

    res = std::is_same<typename dev::execution_space,
                       Kokkos::DefaultHostExecutionSpace>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::memory_space, Kokkos::HostSpace>::value;
    EXPECT_EQ(res, 1);
  }
#if defined(MORPHEUS_ENABLE_SERIAL) && defined(MORPHEUS_ENABLE_OPENMP)
  /* Check if correct execution space is registered if exe and back have
   * different execution spaces
   */
  {
    using exe  = Kokkos::OpenMP;
    using mem  = Kokkos::HostSpace;
    using back = Morpheus::Serial;
    using dev  = Morpheus::Device<exe, mem, back>;

    bool res;

    res = std::is_same<typename dev::backend, back>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::execution_space, exe>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::memory_space, mem>::value;
    EXPECT_EQ(res, 1);
  }
#endif  // MORPHEUS_ENABLE_SERIAL && MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
  /* Check if correct memory space is registered if mem and back have
   * different memory spaces
   */
  {
    using exe  = Kokkos::OpenMP;
    using mem  = Kokkos::CudaSpace;
    using back = Morpheus::Serial;
    using dev  = Morpheus::Device<exe, mem, back>;

    bool res;

    res = std::is_same<typename dev::backend, back>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::execution_space, exe>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename dev::memory_space, mem>::value;
    EXPECT_EQ(res, 1);
  }
#endif  // MORPHEUS_ENABLE_SERIAL || MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ENABLE_CUDA || MORPHEUS_ENABLE_HIP
}

TEST(SpacesTest, HostMirror) {
  {
    using exe    = Kokkos::DefaultHostExecutionSpace;
    using mirror = Morpheus::HostMirror<exe>;
    bool res;

    res = std::is_same<typename mirror::backend,
                       Morpheus::GenericBackend<exe>>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using exe    = Morpheus::DefaultHostExecutionSpace;
    using mirror = Morpheus::HostMirror<exe>;

    bool res;

    res = std::is_same<typename mirror::backend, exe>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using exe    = Morpheus::Generic::DefaultHostExecutionSpace;
    using mirror = Morpheus::HostMirror<exe>;

    bool res;

    res = std::is_same<typename mirror::backend, exe>::value;
    EXPECT_EQ(res, 1);
  }

#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    using exe    = Morpheus::Serial;
    using mirror = Morpheus::HostMirror<exe>;
    if (std::is_same<Morpheus::Serial,
                     Morpheus::DefaultHostExecutionSpace>::value) {
      // keep_exe & keep_mem
      bool res = std::is_same<typename mirror::backend, exe>::value;
      EXPECT_EQ(res, 1);
    } else {
      // keep_mem
      using dev = Morpheus::Device<Kokkos::HostSpace::execution_space,
                                   typename exe::memory_space, exe>;

      bool res = std::is_same<typename mirror::backend, dev>::value;
      EXPECT_EQ(res, 1);
    }
  }

#endif  // MORPHEUS_ENABLE_SERIAL
#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    using exe    = Morpheus::OpenMP;
    using mirror = Morpheus::HostMirror<exe>;

    bool res = std::is_same<typename mirror::backend, exe>::value;
    EXPECT_EQ(res, 1);
  }
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
  {
    using mirror = Morpheus::HostMirror<Morpheus::Cuda>;

    bool res =
        std::is_same<typename mirror::backend, Morpheus::HostSpace>::value;
    EXPECT_EQ(res, 1);
  }
#endif  // MORPHEUS_ENABLE_CUDA

#if defined(MORPHEUS_ENABLE_HIP)
  {
    using mirror = Morpheus::HostMirror<Morpheus::HIP>;

    bool res =
        std::is_same<typename mirror::backend, Morpheus::HostSpace>::value;
    EXPECT_EQ(res, 1);
  }
#endif  // MORPHEUS_ENABLE_HIP
}

#define MORPHEUS_CHECK_BACKEND(REL, SPACE, ref_res)        \
  {                                                        \
    bool _res;                                             \
    _res = Morpheus::REL<Kokkos::SPACE>::value;            \
    EXPECT_EQ(_res, ref_res[0]);                           \
    _res = Morpheus::REL<Morpheus::SPACE>::value;          \
    EXPECT_EQ(_res, ref_res[1]);                           \
    _res = Morpheus::REL<Morpheus::Custom::SPACE>::value;  \
    EXPECT_EQ(_res, ref_res[2]);                           \
    _res = Morpheus::REL<Morpheus::Generic::SPACE>::value; \
    EXPECT_EQ(_res, ref_res[3]);                           \
    /* Checking Alias */                                   \
    _res = Morpheus::REL##_v<Kokkos::SPACE>;               \
    EXPECT_EQ(_res, ref_res[0]);                           \
    _res = Morpheus::REL##_v<Morpheus::SPACE>;             \
    EXPECT_EQ(_res, ref_res[1]);                           \
    _res = Morpheus::REL##_v<Morpheus::Custom::SPACE>;     \
    EXPECT_EQ(_res, ref_res[2]);                           \
    _res = Morpheus::REL##_v<Morpheus::Generic::SPACE>;    \
    EXPECT_EQ(_res, ref_res[3]);                           \
  }

TEST(SpacesTest, HasBackend) {
  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, DefaultHostExecutionSpace,
                           backend_results);
  }

  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, HostSpace, backend_results);
  }

#if defined(MORPHEUS_ENABLE_SERIAL)
  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, Serial, backend_results);
  }
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, OpenMP, backend_results);
  }
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, Cuda, backend_results);
    MORPHEUS_CHECK_BACKEND(has_backend, CudaSpace, backend_results);
  }
#endif  // MORPHEUS_ENABLE_CUDA

#if defined(MORPHEUS_ENABLE_HIP)
  {
    bool backend_results[4] = {0, 1, 1, 1};
    MORPHEUS_CHECK_BACKEND(has_backend, HIP, backend_results);
    MORPHEUS_CHECK_BACKEND(has_backend, HIPSpace, backend_results);
  }
#endif  // MORPHEUS_ENABLE_HIP
}

}  // namespace Test

#endif  // TEST_CORE_TEST_SPACES_HPP
