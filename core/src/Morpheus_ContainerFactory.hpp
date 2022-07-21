/**
 * Morpheus_ContainerFactory.hpp
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

#ifndef MORPHEUS_CONTAINERFACTORY_HPP
#define MORPHEUS_CONTAINERFACTORY_HPP

#include <impl/Morpheus_ContainerFactory.hpp>

namespace Morpheus {

/**
 * \addtogroup other_tags Other Tags
 * \brief Other data structures used to tag data types.
 * \ingroup wrappers_and_tags
 * \{
 */

/**
 * @brief A \p Default tag is used to denote the use of default types.
 *
 */
struct Default {};
/*! \} // end of other_tags group
 */

/**
 * \addtogroup typetraits Type Traits
 * \ingroup utilities
 * \{
 */

/**
 * @brief Checks if the given type \p T is a \p Default type.
 *
 * @tparam T Type passed for check
 */
template <class T>
class is_default {
  typedef char yes[1];
  typedef char no[2];

  template <class U>
  static yes& test(
      U*, typename std::enable_if<std::is_same<U, Default>::value>::type* =
              nullptr);

  template <class U>
  static no& test(...);

 public:
  static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

/**
 * @brief Short-hand to \p is_default.
 *
 * @tparam T Type passed for check.
 */
template <typename T>
inline constexpr bool is_default_v = is_default<T>::value;
/*! \} // end of typetraits group
 */

/**
 * \addtogroup generic_containers Generic Containers
 * \brief Generic containers constructed from the various supported data
 * structures
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief A wrapper that constructs a new container type from \p ContainerType
 using as type arguments the types in \p TypeSet.
 *
 * \par Overview
 * A wrapper that constructs a container type from a set of Type parameters.
 In the case where a template argument is passed as a \p Default this is
 ignored and not passed in the definition of the type. Note that the \p
 Container argument needs to specify a complete container type out of which
 we will be extracting which container that is and define a new type with
 the rest of the template arguments.
 * @brief Generates a \p UnaryContainer from the arguments of the \p Set
 passed.
 *
 * @tparam ContainerType The container type from which the new type will be
 * generated from.
 * @tparam TypeSet A \p Set of types.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  using D = Morpheus::Default; // Short-hand for default tag
 *  using params = Morpheus::Set<float, D, D, Kokkos::Serial>;
 *  using ref = Morpheus::DenseVector<double>;
 *  using Container = Morpheus::UnaryContainer<ref, params>;
 *
 *  using res = Morpheus::DenseVector<float, Kokkos::Serial>;
 *  std::cout << std::is_same<Container, res>::value << std::endl; // prints
 1
 *
 * }
 * \endcode
 */
template <typename ContainerType, typename TypeSet>
struct UnaryContainer {
  using type = typename Impl::UnaryContainerProxy<ContainerType, TypeSet>::type;
};
/*! \} // end of generic_containers group
 */

/**
 * \addtogroup metaprogramming Metaprogramming
 * \ingroup utilities
 * \{
 *
 */

/**
 * @brief Generates a \p TypeList of \p UnaryContainer where each container
 is a
 * specific Morpheus Container type with a variadic number of template
 * arguments.
 *
 * @tparam Container One of Morpheus supported \p Containers
 * @tparam U A type list of all the combination.
 */
template <typename Container, typename U>
struct generate_unary_typelist {
  using type = typename Impl::generate_unary_typelist<Container, U>::type;
};
/*! \} // end of metaprogramming group
 */

}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERFACTORY_HPP