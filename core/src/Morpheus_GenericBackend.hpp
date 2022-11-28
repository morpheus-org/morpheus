/**
 * Morpheus_GenericBackend.hpp
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

#ifndef MORPHEUS_GENERICBACKEND_HPP
#define MORPHEUS_GENERICBACKEND_HPP

#include <Morpheus_SpaceTraits.hpp>

namespace Morpheus {
/**
 * \addtogroup wrappers Wrappers
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
struct GenericBackend {
  static_assert(has_execution_space_v<Space>,
                "Space needs to have a valid Execution Space!");
  static_assert(has_memory_space_v<Space>,
                "Space needs to have a valid Memory Space!");
  using type            = GenericBackend<Space>;
  using backend         = GenericBackend<Space>;
  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using device_type     = Device<execution_space, memory_space, backend>;
};

//!< Namespace for Generic backends to exist in
namespace Generic {
/**
 * @brief A Generic Space that launches kernels in the default Host Space
 *
 */
using DefaultHostExecutionSpace =
    Morpheus::GenericBackend<Kokkos::DefaultHostExecutionSpace>;

/**
 * @brief A Generic Space that launches kernels in the default Space
 *
 */
using DefaultExecutionSpace =
    Morpheus::GenericBackend<Kokkos::DefaultExecutionSpace>;

using HostSpace = Morpheus::GenericBackend<Kokkos::HostSpace>;

#if defined(MORPHEUS_ENABLE_SERIAL)
/**
 * @brief A Generic Space that launches kernels in serial from the performance
 * portable backend (Kokkos)
 *
 */
using Serial = Morpheus::GenericBackend<Kokkos::Serial>;
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
/**
 * @brief A Generic Space that launches kernels in parallel from the
 * performance portable backend (Kokkos) using OpenMP.
 *
 */
using OpenMP = Morpheus::GenericBackend<Kokkos::OpenMP>;
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
/**
 * @brief A Generic Space that launches kernels in parallel from the
 * performance portable backend (Kokkos) using Cuda.
 *
 */
using Cuda      = Morpheus::GenericBackend<Kokkos::Cuda>;
using CudaSpace = Morpheus::GenericBackend<Kokkos::CudaSpace>;
#endif

#if defined(MORPHEUS_ENABLE_HIP)
/**
 * @brief A Generic Space that launches kernels in parallel from the
 * performance portable backend (Kokkos) using HIP.
 *
 */
using HIP      = Morpheus::GenericBackend<Kokkos::HIP>;
using HIPSpace = Morpheus::CustomSpace<Kokkos::HIPSpace>;
#endif
}  // namespace Generic

/*! \} // end of wrappers group
 */

/*! \cond */
namespace Impl {
template <typename T>
struct is_generic_backend_helper : std::false_type {};

template <typename Space>
struct is_generic_backend_helper<GenericBackend<Space>> : std::true_type {};

template <typename T>
using is_generic_backend = typename Impl::is_generic_backend_helper<
    typename std::remove_cv<T>::type>::type;
}  // namespace Impl
/*! \endcond */

/**
 * \addtogroup space_traits Space Traits
 * \ingroup type_traits
 * \{
 *
 */
/**
 * @brief Checks if the given type \p T is a valid generic space i.e is a
 * \p GenericBackend container
 *
 * @tparam T Type passed for check.
 */
template <class T>
class is_generic_backend {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*,
      typename std::enable_if<Impl::is_generic_backend<U>::value ||
                              is_execution_space<U>::value>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};
/**
 * @brief Short-hand to \p is_generic_backend.
 *
 * @tparam T Type passed for check.
 */
template <class T>
inline constexpr bool is_generic_backend_v = is_generic_backend<T>::value;

/*! \} // end of space_traits group
 */

}  // namespace Morpheus

#endif  // MORPHEUS_GENERICBACKEND_HPP