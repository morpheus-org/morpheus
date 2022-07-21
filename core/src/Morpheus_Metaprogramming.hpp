/**
 * Morpheus_Metaprogramming.hpp
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

#ifndef MORPHEUS_METAPROGRAMMING_HPP
#define MORPHEUS_METAPROGRAMMING_HPP

#include <tuple>

namespace Morpheus {

/**
 * \addtogroup metaprogramming Metaprogramming
 * \brief Various metaprogrammes
 * \ingroup utilities
 * \{
 *
 */

/*! \cond */
// Forward declarations
template <typename... Ts>
struct TypeList;
template <typename... Ts>
struct Set;
template <class... Ts>
struct IndexedTypeList;

template <typename T, typename U>
struct cross_product;
template <typename... T>
struct concat;

namespace Impl {
// forward decl
template <typename T, typename U>
struct cross_product;
template <typename... T>
struct concat;

template <typename... Ts, typename... Us>
struct concat<TypeList<Ts...>, TypeList<Us...>> {
  using type = TypeList<Ts..., Us...>;
};

// Partially specialise the empty cases.
template <typename... Us>
struct cross_product<TypeList<>, TypeList<Us...>> {
  using type = TypeList<>;
};

template <typename... Us>
struct cross_product<TypeList<Us...>, TypeList<>> {
  using type = TypeList<>;
};

template <>
struct cross_product<TypeList<>, TypeList<>> {
  using type = TypeList<>;
};

// Generic Case
template <typename T, typename... Ts, typename U, typename... Us>
struct cross_product<TypeList<T, Ts...>, TypeList<U, Us...>> {
  using type = typename concat<
      typename concat<
          TypeList<Set<T, U>>,
          typename cross_product<TypeList<T>, TypeList<Us...>>::type>::type,
      typename cross_product<TypeList<Ts...>, TypeList<U, Us...>>::type>::type;
};

/**
 * @brief Specialization where the first type list contains a Set and the second
 * type list contains a Type
 *
 */
template <typename... T, typename... Ts, typename U, typename... Us>
struct cross_product<TypeList<Set<T...>, Ts...>, TypeList<U, Us...>> {
  using type = typename concat<
      typename concat<TypeList<Set<T..., U>>,
                      typename cross_product<TypeList<Set<T...>>,
                                             TypeList<Us...>>::type>::type,
      typename cross_product<TypeList<Ts...>, TypeList<U, Us...>>::type>::type;
};

/**
 * @brief Specialization where the first type list contains a Type and the
 * second type list contains a Set
 *
 */
template <typename T, typename... Ts, typename... U, typename... Us>
struct cross_product<TypeList<T, Ts...>, TypeList<Set<U...>, Us...>> {
  using type = typename concat<
      typename concat<
          TypeList<Set<T, U...>>,
          typename cross_product<TypeList<T>, TypeList<Us...>>::type>::type,
      typename cross_product<TypeList<Ts...>,
                             TypeList<Set<U...>, Us...>>::type>::type;
};

/**
 * @brief Specialization where the both type lists contains a Set as a first
 * element
 *
 */
template <typename... T, typename... Ts, typename... U, typename... Us>
struct cross_product<TypeList<Set<T...>, Ts...>, TypeList<Set<U...>, Us...>> {
  using type = typename concat<
      typename concat<TypeList<Set<T..., U...>>,
                      typename cross_product<TypeList<Set<T...>>,
                                             TypeList<Us...>>::type>::type,
      typename cross_product<TypeList<Ts...>,
                             TypeList<Set<U...>, Us...>>::type>::type;
};
}  // namespace Impl
/*! \endcond */

/**
 * @brief Compile-time type list
 *
 * @tparam Ts Types passed in the list
 */
template <typename... Ts>
struct TypeList {
  using type = TypeList<Ts...>;
};

/**
 * @brief Compile-time set
 *
 * @tparam Ts Types passed in the set
 */
template <typename... Ts>
struct Set {
  using type = Set<Ts...>;
};

/**
 * @brief Compile-time type list with indexed access
 *
 * @tparam Ts Types passed in the list
 */
template <class... Ts>
struct IndexedTypeList {
  template <std::size_t N>
  using type = typename std::tuple_element<N, std::tuple<Ts...>>::type;
};

/*! \cond */
/**
 * @brief Base case of a \p TypeList of sets.
 */
template <typename... Head_>
struct TypeList<Set<Head_...>> {
  using Head = Set<Head_...>;
  using Tail = void;
};
/*! \endcond */

/**
 * @brief Compile-time linked-list like type list specialisation for when the
 * types passed are \p Set.
 *
 * @tparam Head_ Types included in the first \p Set in the type-list
 * @tparam Tail_ Remaining \p Set types
 */
template <typename... Head_, typename... Tail_>
struct TypeList<Set<Head_...>, Tail_...> {
  using Head = Set<Head_...>;
  using Tail = TypeList<Tail_...>;
};

/**
 * @brief Concatenates types from two \p TypeList in a single \p TypeList.
 *
 * @tparam Ts Types in first type list
 * @tparam Us Types in second type list
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  // Types
 *  struct A {}; struct B {}; struct C {}; struct D {};
 *
 *  // concat<<A, B>, <C, D>> = <A, B, C, D>
 *  using concat_res = typename Morpheus::concat<TypeList<A, B>,
 *                                               TypeList<C, D>>::type;
 *  // reference result
 *  using res = TypeList<A, B, C, D>;
 *
 *  std::cout << std::is_same<concat_res, res>::value << std::endl; // prints 1
 *
 * }
 * \endcode
 */
template <typename... Ts, typename... Us>
struct concat<TypeList<Ts...>, TypeList<Us...>> {
  using type = typename Impl::concat<TypeList<Ts...>, TypeList<Us...>>::type;
};

/**
 * @brief Generates the cross product of the types from two \p TypeList
 *
 * @tparam Ts Types in first type list
 * @tparam Us Types in second type list
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  // Types
 *  struct A {}; struct B {}; struct C {}; struct D {};
 *
 *  // <A, B> x <C, D> = <<A, C>, <A, D>, <B, C>, <B, D>>
 *  using product_res = typename Morpheus::cross_product<TypeList<A, B>,
 *                                                       TypeList<C, D>>::type;
 *  // reference result
 *  using res = TypeList<Set<A,C>, Set<A,D>, Set<B,C>, Set<B,D>>;
 *
 *  std::cout << std::is_same<product_res, res>::value << std::endl; // prints 1
 *
 * }
 * \endcode
 */
template <typename... Ts, typename... Us>
struct cross_product<TypeList<Ts...>, TypeList<Us...>> {
  using type =
      typename Impl::cross_product<TypeList<Ts...>, TypeList<Us...>>::type;
};

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
/*! \}
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
/*! \}
 */

/*! \cond */
namespace Impl {
template <typename... Ts>
struct UnaryContainer {};

template <template <class...> class Container, typename T, typename ValueType>
struct UnaryContainer<Container<T>, ValueType, Default, Default, Default> {
  using type = Container<ValueType>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType>
struct UnaryContainer<Container<T>, ValueType, IndexType, Default, Default> {
  using type = Container<ValueType, IndexType>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename Layout>
struct UnaryContainer<Container<T>, ValueType, Default, Layout, Default> {
  using type = Container<ValueType, Layout>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename Space>
struct UnaryContainer<Container<T>, ValueType, Default, Default, Space> {
  using type = Container<ValueType, Space>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Layout>
struct UnaryContainer<Container<T>, ValueType, IndexType, Layout, Default> {
  using type = Container<ValueType, IndexType, Layout>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Space>
struct UnaryContainer<Container<T>, ValueType, IndexType, Default, Space> {
  using type = Container<ValueType, IndexType, Space>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename Layout, typename Space>
struct UnaryContainer<Container<T>, ValueType, Default, Layout, Space> {
  using type = Container<ValueType, Layout, Space>;
};

template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Layout, typename Space>
struct UnaryContainer<Container<T>, ValueType, IndexType, Layout, Space> {
  using type = Container<ValueType, IndexType, Layout, Space>;
};
}  // namespace Impl

template <typename... Ts>
struct UnaryContainer {};

/*! \endcond */

/**
 * @brief A wrapper that constructs a container type from the template
 * parameters provided.
 *
 * \par Overview
 * A wrapper that constructs a container type from the template
 * parameters provided. In the case where a template argument is passed as a \p
 * Default this is ignored and not passed in the definition of the type. Note
 * that the \p Container argument needs to specify a complete container type out
 * of which we will be extracting which container that is and define a new type
 * with the rest of the template arguments.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  using D = Morpheus::Default; // Short-hand for default tag
 *  using ref = Morpheus::DenseVector<double>;
 *  using Container = Morpheus::UnaryContainer<ref, float, D, D,
 *                                                  Kokkos::Serial>;
 *
 *  // The result type where the default arguments are ignored.
 *  using res = Morpheus::DenseVector<float, Kokkos::Serial>;
 *  std::cout << std::is_same<Container, res>::value << std::endl; // prints 1
 *
 * }
 * \endcode
 *
 * @tparam Container The container type from which the new type will be
 * generated from.
 * @tparam T The required ValueType of Container.
 * @tparam Ts The new types out of which the new container type is created.
 */
template <template <class...> class Container, typename T, typename... Ts>
struct UnaryContainer<Container<T>, Ts...> {
  using type = typename Impl::UnaryContainer<Container<T>, Ts...>::type;
};

/**
 * @brief Specialization of \p UnaryContainer that packs the template arguments
 * as a \p Set.
 *
 * @tparam Container The container type from which the new type will be
 * generated from.
 * @tparam T The required ValueType of Container.
 * @tparam Ts The new types out of which the new container type is created.
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
 *  std::cout << std::is_same<Container, res>::value << std::endl; // prints 1
 *
 * }
 * \endcode
 */
template <template <class...> class Container, typename T, typename... Ts>
struct UnaryContainer<Container<T>, Set<Ts...>> {
  using type = typename UnaryContainer<Container<T>, Ts...>::type;
};

/*! \}
 */
}  // namespace Morpheus

#endif  // MORPHEUS_METAPROGRAMMING_HPP