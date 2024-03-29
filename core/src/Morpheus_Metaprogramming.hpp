/**
 * Morpheus_Metaprogramming.hpp
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

#ifndef MORPHEUS_METAPROGRAMMING_HPP
#define MORPHEUS_METAPROGRAMMING_HPP

#include <impl/Morpheus_Metaprogramming.hpp>

#include <tuple>

namespace Morpheus {

/**
 * \addtogroup metaprogramming Metaprogramming
 * \brief Various metaprogrammes
 * \ingroup utilities
 * \{
 *
 */

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
/*! \} // end of metaprogramming group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_METAPROGRAMMING_HPP