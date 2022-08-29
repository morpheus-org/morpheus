/**
 * Morpheus_GenericSpace.hpp
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

#ifndef MORPHEUS_GENERICSPACE_HPP
#define MORPHEUS_GENERICSPACE_HPP

#include <Morpheus_TypeTraits.hpp>

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
 * @brief A wrapper that converts a valid custom space into a generic one.
 *
 * @tparam Space A space to be converted as a generic.
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
 *  Morpheus::dot<exec>(x, y, z1);  // Dispatches custom implementation
 *
 *  using generic_exec = Morpheus::GenericSpace<exec>;
 *  Morpheus::dot<generic_exec>(x, y, z2);  // Dispatches generic implementation
 *
 * }
 * \endcode
 */
template <typename Space>
struct GenericSpace {
  static_assert(is_execution_space_v<Space>,
                "Space needs to have a valid Execution Space!");
  static_assert(has_memory_space_v<Space>,
                "Space needs to have a valid Memory Space!");
  using generic_space = GenericSpace;
  using type          = Space;

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using device_type     = typename Space::device_type;
};

namespace Generic {

/**
 * @brief A Generic Space that launches kernels in the default Host Space
 *
 */
using DefaultHostExecutionSpace =
    Morpheus::GenericSpace<Kokkos::DefaultHostExecutionSpace>;

/**
 * @brief A Generic Space that launches kernels in the default Space
 *
 */
using DefaultExecutionSpace =
    Morpheus::GenericSpace<Kokkos::DefaultExecutionSpace>;

#if defined(MORPHEUS_ENABLE_SERIAL)
/**
 * @brief A Generic Space that launches kernels in serial from the performance
 * portable backend (Kokkos)
 *
 */
using Serial = Morpheus::GenericSpace<Kokkos::Serial>;
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
/**
 * @brief A Generic Space that launches kernels in parallel from the performance
 * portable backend (Kokkos) using OpenMP.
 *
 */
using OpenMP = Morpheus::GenericSpace<Kokkos::OpenMP>;
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
/**
 * @brief A Generic Space that launches kernels in parallel from the performance
 * portable backend (Kokkos) using Cuda.
 *
 */
using Cuda = Morpheus::GenericSpace<Kokkos::Cuda>;
#endif

#if defined(MORPHEUS_ENABLE_HIP)
/**
 * @brief A Generic Space that launches kernels in parallel from the performance
 * portable backend (Kokkos) using HIP.
 *
 */
using HIP = Morpheus::GenericSpace<Kokkos::Experimental::HIP>;
#endif
}  // namespace Generic

/*! \} // end of wrappers group
 */

/*! \cond */
namespace Impl {
template <typename T>
struct is_generic_space_helper : std::false_type {};

template <typename Space>
struct is_generic_space_helper<GenericSpace<Space>> : std::true_type {};
}  // namespace Impl
/*! \endcond */

/**
 * \addtogroup typetraits Type Traits
 * \ingroup utilities
 * \{
 *
 */
/**
 * @brief Checks if the given type \p T is a valid generic space i.e is a
 * \p GenericSpace container
 *
 * @tparam T Type passed for check.
 */
template <typename T>
using is_generic_space = typename Impl::is_generic_space_helper<
    typename std::remove_cv<T>::type>::type;

/**
 * @brief Short-hand to \p is_generic_space.
 *
 * @tparam T Type passed for check.
 */
template <class T>
inline constexpr bool is_generic_space_v = is_generic_space<T>::value;

/**
 * @brief Checks if the given type \p T has a valid generic space
 *
 * @tparam T Type passed for check.
 *
 * \par Overview
 * For a type to have a generic space we mean that it has a \p generic_space
 * trait that is a valid generic space satisfying the \p is_generic_space check.
 */
template <class T>
class has_generic_space {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<
              is_generic_space_v<typename U::generic_space>>::type* = nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p has_generic_space.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool has_generic_space_v = has_generic_space<T>::value;

/*! \} // end of typetraits group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_GENERICSPACE_HPP