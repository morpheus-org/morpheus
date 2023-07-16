/**
 * Morpheus_ContainerFactory_Impl.hpp
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

#ifndef MORPHEUS_CONTAINERFACTORY_IMPL_HPP
#define MORPHEUS_CONTAINERFACTORY_IMPL_HPP

#include <Morpheus_Metaprogramming.hpp>

/*! \cond */
namespace Morpheus {

// Forward Decl
struct Default;
template <typename T1, typename T2>
struct BinaryContainer;
template <typename ContainerType, typename TypeSet>
struct UnaryContainer;

namespace Impl {
// Forward Decl
template <typename ContainerType, typename TypeSet>
struct UnaryContainerProxy;

template <typename Container, typename T1, typename T2, typename T3,
          typename T4>
struct UnaryContainerProxy_Impl;

// A unary container specialization that is build by selecting the ValueType.
// The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, Default, Default,
                                Default> {
  using type = Container<ValueType>;

  using value_type   = ValueType;
  using index_type   = Default;
  using array_layout = Default;
  using backend      = Default;
};

// A unary container specialization that is build by selecting the ValueType and
// IndexType. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, IndexType, Default,
                                Default> {
  using type = Container<ValueType, IndexType>;

  using value_type   = ValueType;
  using index_type   = IndexType;
  using array_layout = Default;
  using backend      = Default;
};

// A unary container specialization that is build by selecting the ValueType and
// Layout. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename Layout>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, Default, Layout,
                                Default> {
  using type = Container<ValueType, Layout>;

  using value_type   = ValueType;
  using index_type   = Default;
  using array_layout = Layout;
  using backend      = Default;
};

// A unary container specialization that is build by selecting the ValueType and
// Space. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename Space>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, Default, Default,
                                Space> {
  using type = Container<ValueType, Space>;

  using value_type   = ValueType;
  using index_type   = Default;
  using array_layout = Default;
  using backend      = Space;
};

// A unary container specialization that is build by selecting the ValueType,
// IndexType and Layout. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Layout>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, IndexType, Layout,
                                Default> {
  using type = Container<ValueType, IndexType, Layout>;

  using value_type   = ValueType;
  using index_type   = IndexType;
  using array_layout = Layout;
  using backend      = Default;
};

// A unary container specialization that is build by selecting the ValueType,
// IndexType and Space. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Space>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, IndexType, Default,
                                Space> {
  using type = Container<ValueType, IndexType, Space>;

  using value_type   = ValueType;
  using index_type   = IndexType;
  using array_layout = Default;
  using backend      = Space;
};

// A unary container specialization that is build by selecting the ValueType,
// Layout and Space. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename Layout, typename Space>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, Default, Layout,
                                Space> {
  using type = Container<ValueType, Layout, Space>;

  using value_type   = ValueType;
  using index_type   = Default;
  using array_layout = Layout;
  using backend      = Space;
};

// A unary container specialization that is build by selecting the ValueType,
// IndexType, Layout and Space. The rest of the parameters are set to default.
template <template <class...> class Container, typename T, typename ValueType,
          typename IndexType, typename Layout, typename Space>
struct UnaryContainerProxy_Impl<Container<T>, ValueType, IndexType, Layout,
                                Space> {
  using type = Container<ValueType, IndexType, Layout, Space>;

  using value_type   = ValueType;
  using index_type   = IndexType;
  using array_layout = Layout;
  using backend      = Space;
};

// Takes in types as a set and forwards it to the UnaryContainer.
template <template <class...> class Container, typename T, typename T1,
          typename T2, typename T3, typename T4>
struct UnaryContainerProxy<Container<T>, Set<T1, T2, T3, T4>> {
  using unary = UnaryContainerProxy_Impl<Container<T>, T1, T2, T3, T4>;
  using type  = typename unary::type;

  using value_type   = typename unary::value_type;
  using index_type   = typename unary::index_type;
  using array_layout = typename unary::array_layout;
  using backend      = typename unary::backend;
};

template <typename... Ts>
struct generate_unary_typelist {};

// Partially specialise the empty cases.
template <typename T>
struct generate_unary_typelist<TypeList<>, T> {
  using type = TypeList<>;
};

template <typename T>
struct generate_unary_typelist<T, TypeList<>> {
  using type = TypeList<>;
};

// Generate unary container from a Set of parameter types
template <template <typename...> class Container, typename T, typename... U>
struct generate_unary_typelist<Container<T>, Set<U...>> {
  using type = UnaryContainer<Container<T>, Set<U...>>;
};

// Generate unary container
template <template <typename...> class Container, typename T, typename... U,
          typename... Us>
struct generate_unary_typelist<Container<T>, TypeList<Set<U...>, Us...>> {
  using type = typename concat<
      TypeList<typename generate_unary_typelist<Container<T>, Set<U...>>::type>,
      typename generate_unary_typelist<Container<T>,
                                       TypeList<Us...>>::type>::type;
};

template <typename... Ts>
struct generate_binary_typelist {};

// Generate binary container from a Set of containers
template <typename T1, typename T2>
struct generate_binary_typelist<Set<T1, T2>> {
  using type = TypeList<BinaryContainer<T1, T2>>;
};

// Generate binary container from the base case i.e only one element in TypeList
template <typename S>
struct generate_binary_typelist<TypeList<S>> {
  using type = typename Impl::generate_binary_typelist<S>::type;
};

// Generate binary container by processing each element in TypeList
template <typename S, typename... Ss>
struct generate_binary_typelist<TypeList<S, Ss...>> {
  using type = typename concat<
      typename Impl::generate_binary_typelist<S>::type,
      typename Impl::generate_binary_typelist<TypeList<Ss...>>::type>::type;
};

template <typename... Lists>
struct generate_binary_typelist_proxy {};

// Process the two TypeLists and generate all possible combinations of the two.
template <typename... List1, typename... List2>
struct generate_binary_typelist_proxy<TypeList<List1...>, TypeList<List2...>> {
  using type =
      typename Impl::generate_binary_typelist<typename Morpheus::cross_product<
          TypeList<List1...>, TypeList<List2...>>::type>::type;
};

template <typename... Ts>
struct mirror_params;

template <template <typename V1, typename... PS1> typename Src, typename V1,
          typename... PS1, template <typename V2, typename... PS2> typename Dst,
          typename V2, typename... PS2>
struct mirror_params<Src<V1, PS1...>, Dst<V2, PS2...>> {
  using type = Dst<V1, PS1...>;
};

}  // namespace Impl
}  // namespace Morpheus
/*! \endcond */
#endif  // MORPHEUS_CONTAINERFACTORY_IMPL_HPP