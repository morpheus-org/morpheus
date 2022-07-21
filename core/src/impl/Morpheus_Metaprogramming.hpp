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

#ifndef MORPHEUS_IMPL_METAPROGRAMMING_HPP
#define MORPHEUS_IMPL_METAPROGRAMMING_HPP

/*! \cond */
namespace Morpheus {

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
}  // namespace Morpheus
/*! \endcond */
#endif  // MORPHEUS_IMPL_METAPROGRAMMING_HPP