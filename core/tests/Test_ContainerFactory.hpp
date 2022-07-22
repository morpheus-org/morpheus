/**
 * Test_ContainerFactory.hpp
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

#ifndef TEST_CORE_TEST_CONTAINERFACTORY_HPP
#define TEST_CORE_TEST_CONTAINERFACTORY_HPP

#include <Morpheus_Core.hpp>

namespace Test {

template <typename T, typename... Ts>
struct Container1 {};
template <typename T, typename... Ts>
struct Container2 {};

/**
 * @brief Checks if a type is a \p Default.
 *
 */
TEST(ContainerFactoryTest, IsDefault) {
  bool res = Morpheus::is_default<Morpheus::Default>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_default<double>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_default<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_default<Morpheus::Impl::SparseMatrixTag>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_default_v<Morpheus::Default>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_default_v<double>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief Checks the compile-time type generation of a \p UnaryContainer from a
 * sample container and a set of type arguments. In other word, we expect the
 * unary container to adopt the provided container type with the new set of
 * parameters.
 *
 */
TEST(ContainerFactoryTest, UnaryContainer) {
  using DD = Morpheus::Default;
  struct t {};
  struct A {};
  struct B {};
  struct C {};
  struct D {};

  {
    using set       = Morpheus::Set<A, DD, DD, DD>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, B, DD, DD>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, DD, B, DD>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, DD, DD, B>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, B, C, DD>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B, C>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, B, DD, C>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B, C>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, DD, B, C>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B, C>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using set       = Morpheus::Set<A, B, C, D>;
    using unary     = Morpheus::UnaryContainer<Container1<t>, set>;
    using reference = Container1<A, B, C, D>;

    bool res = std::is_same<typename unary::type, reference>::value;
    EXPECT_EQ(res, 1);
  }
}

template <typename... Ts>
struct to_gtest_types {};

template <typename... Ts>
struct to_gtest_types<Morpheus::TypeList<Ts...>> {
  using type = ::testing::Types<Ts...>;
};

/**
 * @brief Generate a list of unary types that emerge from all the possible
 * combinations of type sets generated.
 *
 */
TEST(ContainerFactoryTest, GenerateUnaryTypeList) {
  using DD = Morpheus::Default;
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};
  struct G {};
  struct H {};

  using value_tlist  = Morpheus::TypeList<A, B, C>;
  using index_tlist  = Morpheus::TypeList<D, E, DD>;
  using layout_tlist = Morpheus::TypeList<F, G, DD>;
  using space_tlist  = Morpheus::TypeList<H, DD>;

  using types_set = typename Morpheus::cross_product<
      value_tlist,
      typename Morpheus::cross_product<
          index_tlist, typename Morpheus::cross_product<
                           layout_tlist, space_tlist>::type>::type>::type;

  using unary_types =
      typename Morpheus::generate_unary_typelist<Container1<double>,
                                                 types_set>::type;

  using res_t = Morpheus::TypeList<
      Container1<A, D, F, H>, Container1<A, D, F>, Container1<A, D, G, H>,
      Container1<A, D, G>, Container1<A, D, H>, Container1<A, D>,
      Container1<A, E, F, H>, Container1<A, E, F>, Container1<A, E, G, H>,
      Container1<A, E, G>, Container1<A, E, H>, Container1<A, E>,
      Container1<A, F, H>, Container1<A, F>, Container1<A, G, H>,
      Container1<A, G>, Container1<A, H>, Container1<A>, Container1<B, D, F, H>,
      Container1<B, D, F>, Container1<B, D, G, H>, Container1<B, D, G>,
      Container1<B, D, H>, Container1<B, D>, Container1<B, E, F, H>,
      Container1<B, E, F>, Container1<B, E, G, H>, Container1<B, E, G>,
      Container1<B, E, H>, Container1<B, E>, Container1<B, F, H>,
      Container1<B, F>, Container1<B, G, H>, Container1<B, G>, Container1<B, H>,
      Container1<B>, Container1<C, D, F, H>, Container1<C, D, F>,
      Container1<C, D, G, H>, Container1<C, D, G>, Container1<C, D, H>,
      Container1<C, D>, Container1<C, E, F, H>, Container1<C, E, F>,
      Container1<C, E, G, H>, Container1<C, E, G>, Container1<C, E, H>,
      Container1<C, E>, Container1<C, F, H>, Container1<C, F>,
      Container1<C, G, H>, Container1<C, G>, Container1<C, H>, Container1<C>>;

  bool res = std::is_same<unary_types, res_t>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Generate a list of unary types that emerge from all the possible
 * combinations of type sets generated - each type list holds only one type so
 * we expect only one set to be generated and passed as a type set.
 *
 */
TEST(ContainerFactoryTest, GenerateUnaryTypeListSingleEntry) {
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};
  struct G {};
  struct H {};

  using value_tlist  = Morpheus::TypeList<A>;
  using index_tlist  = Morpheus::TypeList<D>;
  using layout_tlist = Morpheus::TypeList<F>;
  using space_tlist  = Morpheus::TypeList<H>;

  using types_set = typename Morpheus::cross_product<
      value_tlist,
      typename Morpheus::cross_product<
          index_tlist, typename Morpheus::cross_product<
                           layout_tlist, space_tlist>::type>::type>::type;

  using unary_types =
      typename Morpheus::generate_unary_typelist<Container1<double>,
                                                 types_set>::type;

  using res_t = Morpheus::TypeList<Container1<A, D, F, H>>;

  bool res = std::is_same<unary_types, res_t>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Generate a list of unary types that emerge from all the possible
 * combinations of type sets generated - an empty type set is passed to generate
 * the unary type list so we expect the result to be an empty type list.
 *
 */
TEST(ContainerFactoryTest, GenerateUnaryTypeListNoEntry) {
  using empty_list = Morpheus::TypeList<>;

  using unary_types =
      typename Morpheus::generate_unary_typelist<Container1<double>,
                                                 empty_list>::type;

  using res_t = empty_list;

  bool res = std::is_same<unary_types, res_t>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Checks the compile-time type generation of a \p BinaryContainer from
 * two sample containers. We expect the binary container to have \p type1 and \p
 * type2 same to the types of the containers passed each time.
 *
 */
TEST(ContainerFactoryTest, BinaryContainer) {
  struct A {};
  struct B {};
  struct C {};
  struct D {};

  {
    using c1  = Container1<A, B>;
    using c2  = Container2<C, D>;
    using bin = Morpheus::BinaryContainer<c1, c2>;

    bool res = std::is_same<typename bin::type1, c1>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename bin::type2, c2>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using c1  = Container1<A, B>;
    using c2  = Container2<C, D>;
    using bin = Morpheus::BinaryContainer<c2, c1>;

    bool res = std::is_same<typename bin::type1, c2>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename bin::type2, c1>::value;
    EXPECT_EQ(res, 1);
  }

  {
    using c1  = Container1<A, B>;
    using bin = Morpheus::BinaryContainer<c1, c1>;

    bool res = std::is_same<typename bin::type1, c1>::value;
    EXPECT_EQ(res, 1);
    res = std::is_same<typename bin::type2, c1>::value;
    EXPECT_EQ(res, 1);
  }
}

/**
 * @brief Generate a list of binary types that emerge from all the possible
 * combinations of two unary type lists.
 *
 */
TEST(ContainerFactoryTest, GenerateBinaryTypeList) {
  using DD = Morpheus::Default;
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};
  struct G {};
  struct H {};

  using value_tlist  = Morpheus::TypeList<A>;
  using index_tlist  = Morpheus::TypeList<D, DD>;
  using layout_tlist = Morpheus::TypeList<F, DD>;
  using space_tlist  = Morpheus::TypeList<H, DD>;

  using types_set = typename Morpheus::cross_product<
      value_tlist,
      typename Morpheus::cross_product<
          index_tlist, typename Morpheus::cross_product<
                           layout_tlist, space_tlist>::type>::type>::type;

  using container1_types =
      typename Morpheus::generate_unary_typelist<Container1<double>,
                                                 types_set>::type;

  using res_container1_t =
      Morpheus::TypeList<Container1<A, D, F, H>, Container1<A, D, F>,
                         Container1<A, D, H>, Container1<A, D>,
                         Container1<A, F, H>, Container1<A, F>,
                         Container1<A, H>, Container1<A>>;

  bool res = std::is_same<container1_types, res_container1_t>::value;
  EXPECT_EQ(res, 1);

  using container2_types =
      typename Morpheus::generate_unary_typelist<Container2<double>,
                                                 types_set>::type;

  using res_container2_t =
      Morpheus::TypeList<Container2<A, D, F, H>, Container2<A, D, F>,
                         Container2<A, D, H>, Container2<A, D>,
                         Container2<A, F, H>, Container2<A, F>,
                         Container2<A, H>, Container2<A>>;

  res = std::is_same<container2_types, res_container2_t>::value;
  EXPECT_EQ(res, 1);
  std::cout << Morpheus::is_container<Container1<A, D, F, H>>::value
            << std::endl;
  using binary_containers =
      typename Morpheus::generate_binary_typelist<container1_types,
                                                  container2_types>::type;
  using res_t = Morpheus::TypeList<
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F, H>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, D, F>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, D, H>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, D>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, F, H>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, F>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A, H>, Container2<A>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, D, F, H>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, D, F>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, D, H>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, D>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, F, H>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, F>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A, H>>,
      Morpheus::BinaryContainer<Container1<A>, Container2<A>>>;

  res = std::is_same<binary_containers, res_t>::value;
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_METAPROGRAMMING_HPP
