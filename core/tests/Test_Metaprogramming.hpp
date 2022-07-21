/**
 * Test_Metaprogramming.hpp
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

#ifndef TEST_CORE_TEST_METAPROGRAMMING_HPP
#define TEST_CORE_TEST_METAPROGRAMMING_HPP

#include <Morpheus_Core.hpp>

namespace Test {
/**
 * @brief Checks the compile-time concatenation of two type lists
 *
 */
TEST(MetaprogrammingTest, Concat) {
  // Types
  struct A {};
  struct B {};
  struct C {};
  struct D {};

  // concat<<A, B>, <C, D>> = <A, B, C, D>
  using concat_res1 = typename Morpheus::concat<Morpheus::TypeList<A, B>,
                                                Morpheus::TypeList<C, D>>::type;

  bool res = std::is_same<concat_res1, Morpheus::TypeList<A, B, C, D>>::value;
  EXPECT_EQ(res, 1);

  // concat<<A, B>, <C, D, C, D>> = <A, B, C, D, C, D>
  using concat_res2 =
      typename Morpheus::concat<Morpheus::TypeList<A, B>,
                                Morpheus::TypeList<C, D, C, D>>::type;

  res = std::is_same<concat_res2, Morpheus::TypeList<A, B, C, D, C, D>>::value;
  EXPECT_EQ(res, 1);

  // concat<<A, B, A, B>, <C, D>> = <A, B, A, B, C, D>
  using concat_res3 = typename Morpheus::concat<Morpheus::TypeList<A, B, A, B>,
                                                Morpheus::TypeList<C, D>>::type;

  res = std::is_same<concat_res3, Morpheus::TypeList<A, B, A, B, C, D>>::value;
  EXPECT_EQ(res, 1);

  // concat<<<A>, B>, <C, D>> = <<A>, B, C, D>
  using concat_res4 =
      typename Morpheus::concat<Morpheus::TypeList<Morpheus::TypeList<A>, B>,
                                Morpheus::TypeList<C, D>>::type;

  res = std::is_same<concat_res4,
                     Morpheus::TypeList<Morpheus::TypeList<A>, B, C, D>>::value;
  EXPECT_EQ(res, 1);

  // concat<<<A>, B>, <C, <D>>> = <<A>, B, C, <D>>
  using concat_res5 = typename Morpheus::concat<
      Morpheus::TypeList<Morpheus::TypeList<A>, B>,
      Morpheus::TypeList<C, Morpheus::TypeList<D>>>::type;

  res = std::is_same<concat_res5,
                     Morpheus::TypeList<Morpheus::TypeList<A>, B, C,
                                        Morpheus::TypeList<D>>>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Checks the compile-time cross product of two type lists of types
 *
 */
TEST(MetaprogrammingTest, CrossProduct) {
  // Types
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};

  // <A, B> x <C, D> = <<A, C>, <A, D>, <B, C>, <B, D>>
  using product_res1 =
      typename Morpheus::cross_product<Morpheus::TypeList<A, B>,
                                       Morpheus::TypeList<C, D>>::type;

  bool res = std::is_same<
      product_res1,
      Morpheus::TypeList<Morpheus::Set<A, C>, Morpheus::Set<A, D>,
                         Morpheus::Set<B, C>, Morpheus::Set<B, D>>>::value;
  EXPECT_EQ(res, 1);

  // <A> x <B, C, D> = <<A, B>, <A, C>, <A, D>>
  using product_res2 =
      typename Morpheus::cross_product<Morpheus::TypeList<A>,
                                       Morpheus::TypeList<B, C, D>>::type;

  res =
      std::is_same<product_res2,
                   Morpheus::TypeList<Morpheus::Set<A, B>, Morpheus::Set<A, C>,
                                      Morpheus::Set<A, D>>>::value;
  EXPECT_EQ(res, 1);

  // <A, B, C> x <D> = <<A, D>, <B, D>, <C, D>>
  using product_res3 =
      typename Morpheus::cross_product<Morpheus::TypeList<A, B, C>,
                                       Morpheus::TypeList<D>>::type;

  res =
      std::is_same<product_res3,
                   Morpheus::TypeList<Morpheus::Set<A, D>, Morpheus::Set<B, D>,
                                      Morpheus::Set<C, D>>>::value;
  EXPECT_EQ(res, 1);

  // <> x <A, B, C, D> = <>
  using product_res4 =
      typename Morpheus::cross_product<Morpheus::TypeList<>,
                                       Morpheus::TypeList<A, B, C, D>>::type;

  res = std::is_same<product_res4, Morpheus::TypeList<>>::value;
  EXPECT_EQ(res, 1);

  // <A, B, C, D> x <> = <>
  using product_res5 =
      typename Morpheus::cross_product<Morpheus::TypeList<A, B, C, D>,
                                       Morpheus::TypeList<>>::type;

  res = std::is_same<product_res5, Morpheus::TypeList<>>::value;
  EXPECT_EQ(res, 1);

  // <> x <> = <>
  using product_res6 =
      typename Morpheus::cross_product<Morpheus::TypeList<>,
                                       Morpheus::TypeList<>>::type;

  res = std::is_same<product_res6, Morpheus::TypeList<>>::value;
  EXPECT_EQ(res, 1);

  // <A, B, C> x <D, E, F> = <<A, D>, <A, E>, <A, F>,
  //                          <B, D>, <B, E>, <B, F>,
  //                          <C, D>, <C, E>, <C, F>>
  using product_res7 =
      typename Morpheus::cross_product<Morpheus::TypeList<A, B, C>,
                                       Morpheus::TypeList<D, E, F>>::type;

  res =
      std::is_same<product_res7,
                   Morpheus::TypeList<Morpheus::Set<A, D>, Morpheus::Set<A, E>,
                                      Morpheus::Set<A, F>, Morpheus::Set<B, D>,
                                      Morpheus::Set<B, E>, Morpheus::Set<B, F>,
                                      Morpheus::Set<C, D>, Morpheus::Set<C, E>,
                                      Morpheus::Set<C, F>>>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Checks the compile-time cross product of many type lists of
 * types. Each product is a separate result type which is then propagated to the
 * next.
 *
 * e.g <A,B> x <C,D> x <E,F>
 *      = <A, B> x <[C, E], [C, F], [D, E], [D, F]>
 *      = <[A, C, E], [A, C, F], [A, D, E], [A, D, F]
 *         [B, C, E], [B, C, F], [B, D, E], [B, D, F]>
 *
 */
TEST(MetaprogrammingTest, MultipleCrossProducts) {
  // Types
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};

  // <A,B> x <C,D> x <E,F>
  // = <set<A,C>, set<A,D>, set<B,C>, set<B,D>> x <E, F>
  // = <set<A,C,E>, set<A,C,F>, set<A,D,E>, set<A,D,F>,
  // 	set<B,C,E>, set<B,C,F>, set<B,D,E>, set<B,D,F>>

  // <A,B> x <C,D>
  // = <set<A,C>, set<A,D>, set<B,C>, set<B,D>>
  using cross1 =
      typename Morpheus::cross_product<Morpheus::TypeList<A, B>,
                                       Morpheus::TypeList<C, D>>::type;
  // <set<A,C>, set<A,D>, set<B,C>, set<B,D>> x <E, F>
  // = <set<A,C,E>, set<A,C,F>, set<A,D,E>, set<A,D,F>,
  // 	set<B,C,E>, set<B,C,F>, set<B,D,E>, set<B,D,F>>
  using cross2 =
      typename Morpheus::cross_product<cross1, Morpheus::TypeList<E, F>>::type;

  bool res = std::is_same<
      cross2, Morpheus::TypeList<Morpheus::Set<A, C, E>, Morpheus::Set<A, C, F>,
                                 Morpheus::Set<A, D, E>, Morpheus::Set<A, D, F>,
                                 Morpheus::Set<B, C, E>, Morpheus::Set<B, C, F>,
                                 Morpheus::Set<B, D, E>,
                                 Morpheus::Set<B, D, F>>>::value;
  EXPECT_EQ(res, 1);

  // <set<A,C>, set<A,D>, set<B,C>, set<B,D>> x <set<E, F>>
  // = <set<A,C,E,F>, set<A,D,E,F>,
  //    set<B,C,E,F>, set<B,D,E,F>>
  using cross3 = typename Morpheus::cross_product<
      cross1, Morpheus::TypeList<Morpheus::Set<E, F>>>::type;

  res = std::is_same<
      cross3, Morpheus::TypeList<
                  Morpheus::Set<A, C, E, F>, Morpheus::Set<A, D, E, F>,
                  Morpheus::Set<B, C, E, F>, Morpheus::Set<B, D, E, F>>>::value;
  EXPECT_EQ(res, 1);
}

/**
 * @brief Checks the compile-time cross product of many type lists of
 * types using a nested cross product i.e we only get the final type.
 *
 * e.g <A,B> x <C,D> x <E,F>
 *      = <[A, C, E], [A, C, F], [A, D, E], [A, D, F]
 *         [B, C, E], [B, C, F], [B, D, E], [B, D, F]>
 *
 */
TEST(MetaprogrammingTest, NestedCrossProduct) {
  // Types
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};
  using T1 = Morpheus::TypeList<A, B>;
  using T2 = Morpheus::TypeList<C, D>;
  using T3 = Morpheus::TypeList<E, F>;

  // Nested cross product
  using types_set = typename Morpheus::cross_product<
      T1, typename Morpheus::cross_product<T2, T3>::type>::type;
  using ref_set =
      Morpheus::TypeList<Morpheus::Set<A, C, E>, Morpheus::Set<A, C, F>,
                         Morpheus::Set<A, D, E>, Morpheus::Set<A, D, F>,
                         Morpheus::Set<B, C, E>, Morpheus::Set<B, C, F>,
                         Morpheus::Set<B, D, E>, Morpheus::Set<B, D, F>>;
  bool res = std::is_same<types_set, ref_set>::value;
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_METAPROGRAMMING_HPP
