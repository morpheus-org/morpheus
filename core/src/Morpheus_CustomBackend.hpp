/**
 * Morpheus_CustomBackend.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_CUSTOMBACKEND_HPP
#define MORPHEUS_CUSTOMBACKEND_HPP

#include <Morpheus_SpaceTraits.hpp>
#include <fwd/Morpheus_Fwd_Spaces.hpp>

namespace Morpheus {
/**
 * \defgroup wrappers_and_tags Wrappers and Tags
 * \par Overview
 * TODO
 */
/**
 * \addtogroup wrappers Wrappers
 * \brief Data structures used to wrap around data types
 * \ingroup wrappers_and_tags
 * \{
 */

/**
 * @brief A wrapper that converts a valid space into a custom backend.
 *
 * @tparam Space A space to be converted as a custom.
 *
 * \par Overview
 * A wrapper like that is helpful if we want to distinguish algorithms that
 * explicitly use a custom backend from the ones that we want to use a
 * performance portable kernel and effectively keep a single source
 * implementation.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  Morpheus::DenseVector<double, Kokkos::HostSpace> x(10, 5.0), y(10, 3.0);
 *  Morpheus::DenseVector<double, Kokkos::HostSpace> z1(10,0.0), z2(10,0.0);
 *
 *  using exec = Kokkos::DefaultHostSpace;
 *  Morpheus::dot<exec>(x, y, z2);  // Dispatches generic implementation
 *
 *  using custom_back = Morpheus::CustomBackend<exec>;
 *  Morpheus::dot<custom_back>(x, y, z1);  // Dispatches custom implementation
 *
 *  using generic_back = Morpheus::GenericBackend<exec>;
 *  Morpheus::dot<generic_back>(x, y, z2);  // Dispatches generic implementation
 *
 * }
 * \endcode
 */
template <typename Space>
struct CustomBackend {
  static_assert(has_execution_space_v<Space>,
                "Space needs to have a valid Execution Space!");
  static_assert(has_memory_space_v<Space>,
                "Space needs to have a valid Memory Space!");
  using backend = CustomBackend<Space>;  //!< The type of the custom backend
  using execution_space =
      typename Space::execution_space;  //!< Execution space of custom backend
  using memory_space =
      typename Space::memory_space;  //!< Memory space of the custom backend
  using device_type = Device<execution_space, memory_space,
                             backend>;  //!< Device type of custom backend
};

/**
 * @namespace Morpheus::Custom
 * @brief Namespace for Custom Backend definitions
 */
namespace Custom {
/*! @brief A Custom Space that launches kernels in the default Host Space
 */
using DefaultHostExecutionSpace =
    Morpheus::CustomBackend<Kokkos::DefaultHostExecutionSpace>;
/*! @brief A Custom Space that launches kernels in the default Space
 */
using DefaultExecutionSpace =
    Morpheus::CustomBackend<Kokkos::DefaultExecutionSpace>;
/*! @brief The Custom Host memory space
 */
using HostSpace = Morpheus::CustomBackend<Kokkos::HostSpace>;

#if defined(MORPHEUS_ENABLE_SERIAL)
/*! @brief A Custom Space that launches kernels in serial using the Serial
 * backend
 */
using Serial = Morpheus::CustomBackend<Kokkos::Serial>;

#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
/*! @brief A Custom Space that launches kernels in parallel using the OpenMP
 * backend.
 */
using OpenMP = Morpheus::CustomBackend<Kokkos::OpenMP>;

#endif

#if defined(MORPHEUS_ENABLE_CUDA)
/*! @brief A Custom Space that launches kernels in parallel using the Cuda
 * backend.
 */
using Cuda = Morpheus::CustomBackend<Kokkos::Cuda>;
/*! @brief The Custom Cuda memory space
 */
using CudaSpace = Morpheus::CustomBackend<Kokkos::CudaSpace>;
#endif

#if defined(MORPHEUS_ENABLE_HIP)
/*! @brief A Custom Space that launches kernels in parallel using the HIP
 * backend.
 */
using HIP = Morpheus::CustomBackend<Kokkos::HIP>;

/*! @brief The Custom HIP memory space
 */
using HIPSpace = Morpheus::CustomBackend<Kokkos::HIPSpace>;
#endif
}  // namespace Custom

/*! @brief Alias to \p Morpheus::Custom::DefaultHostExecutionSpace
 */
using DefaultHostExecutionSpace = Custom::DefaultHostExecutionSpace;
/*! @brief Alias to \p Morpheus::Custom::DefaultExecutionSpace
 */
using DefaultExecutionSpace = Custom::DefaultExecutionSpace;
/*! @brief Alias to \p Morpheus::Custom::HostSpace
 */
using HostSpace = Custom::HostSpace;

#if defined(MORPHEUS_ENABLE_SERIAL)
/*! @brief Alias to \p Morpheus::Custom::Serial
 */
using Serial = Custom::Serial;
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
/*! @brief Alias to \p Morpheus::Custom::OpenMP
 */
using OpenMP = Custom::OpenMP;
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
/*! @brief Alias to \p Morpheus::Custom::Cuda
 */
using Cuda = Custom::Cuda;
/*! @brief Alias to \p Morpheus::Custom::CudaSpace
 */
using CudaSpace = Custom::CudaSpace;
#endif

#if defined(MORPHEUS_ENABLE_HIP)
/*! @brief Alias to \p Morpheus::Custom::HIP
 */
using HIP = Custom::HIP;
/*! @brief Alias to \p Morpheus::Custom::HIPSpace
 */
using HIPSpace = Custom::HIPSpace;
#endif

/*! \} // end of wrappers group
 */

/*! \cond */
namespace Impl {
template <typename T>
struct is_custom_backend_helper : std::false_type {};

template <typename Space>
struct is_custom_backend_helper<CustomBackend<Space>> : std::true_type {};
}  // namespace Impl
/*! \endcond */

/**
 * \addtogroup space_traits Space Traits
 * \ingroup type_traits
 * \{
 *
 */
/**
 * @brief Checks if the given type \p T is a valid custom backend i.e is a
 * \p CustomBackend container.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
using is_custom_backend = typename Impl::is_custom_backend_helper<
    typename std::remove_cv<T>::type>::type;

/**
 * @brief Short-hand to \p is_custom_backend.
 *
 * @tparam T Type passed for check.
 */
template <class T>
inline constexpr bool is_custom_backend_v = is_custom_backend<T>::value;

/*! \} // end of space_traits group
 */

}  // namespace Morpheus

#endif  // MORPHEUS_CUSTOMBACKEND_HPP