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
#define VALIDATE_UNARY_CONTAINER(r, NewType, RefType, RefValueType,            \
                                 RefIndexType, RefLayout, RefSpace)            \
  {                                                                            \
    r = std::is_same<typename NewType::type, RefType>::value;                  \
    EXPECT_EQ(r, 1);                                                           \
    r = std::is_same<typename NewType::type::value_type, RefValueType>::value; \
    EXPECT_EQ(r, 1);                                                           \
    r = std::is_same<typename NewType::type::index_type, RefIndexType>::value; \
    EXPECT_EQ(r, 1);                                                           \
    r = std::is_same<typename NewType::type::array_layout, RefLayout>::value;  \
    EXPECT_EQ(r, 1);                                                           \
    r = std::is_same<typename NewType::type::execution_space,                  \
                     RefSpace>::value;                                         \
    EXPECT_EQ(r, 1);                                                           \
  }

/**
 * @brief Checks the compile-time type generation of a \p DenseVector container
 * from a nubmer of template arguments.
 *
 */
TEST(ContainerFactoryTest, UnaryContainer_DenseVector) {
  using D   = Morpheus::Default;
  using f   = float;
  using i   = long long;
  using l   = Kokkos::LayoutLeft;
  using s   = Kokkos::Serial;
  using con = Morpheus::DenseVector<double>;

  using def_space  = Kokkos::DefaultExecutionSpace;
  using def_layout = Kokkos::LayoutRight;
  bool res;

  using U_v     = Morpheus::UnaryContainer<con, f, D, D, D>;
  using ref_U_v = Morpheus::DenseVector<f>;
  VALIDATE_UNARY_CONTAINER(res, U_v, ref_U_v, f, int, def_layout, def_space);

  using U_vi     = Morpheus::UnaryContainer<con, f, i, D, D>;
  using ref_U_vi = Morpheus::DenseVector<f, i>;
  VALIDATE_UNARY_CONTAINER(res, U_vi, ref_U_vi, f, i, def_layout, def_space);

  using U_vl     = Morpheus::UnaryContainer<con, f, D, l, D>;
  using ref_U_vl = Morpheus::DenseVector<f, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vl, ref_U_vl, f, int, l, def_space);

  using U_vs     = Morpheus::UnaryContainer<con, f, D, D, s>;
  using ref_U_vs = Morpheus::DenseVector<f, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vs, ref_U_vs, f, int,
                           typename s::array_layout, s);

  using U_vil     = Morpheus::UnaryContainer<con, f, i, l, D>;
  using ref_U_vil = Morpheus::DenseVector<f, i, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vil, ref_U_vil, f, i, l, def_space);

  using U_vis     = Morpheus::UnaryContainer<con, f, i, D, s>;
  using ref_U_vis = Morpheus::DenseVector<f, i, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vis, ref_U_vis, f, i, def_layout, s);

  using U_vls     = Morpheus::UnaryContainer<con, f, D, l, s>;
  using ref_U_vls = Morpheus::DenseVector<f, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vls, ref_U_vls, f, int, l, s);

  using U_vils     = Morpheus::UnaryContainer<con, f, i, l, s>;
  using ref_U_vils = Morpheus::DenseVector<f, i, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vils, ref_U_vils, f, i, l, s);
}

/**
 * @brief Checks the compile-time type generation of a \p DenseVector container
 * from a set of template arguments.
 *
 */
TEST(ContainerFactoryTest, UnaryContainer_DenseVector_FromSet) {
  using D   = Morpheus::Default;
  using f   = float;
  using i   = long long;
  using l   = Kokkos::LayoutLeft;
  using s   = Kokkos::Serial;
  using con = Morpheus::DenseVector<double>;

  using def_space  = Kokkos::DefaultExecutionSpace;
  using def_layout = Kokkos::LayoutRight;
  bool res;

  using v       = Morpheus::Set<f, D, D, D>;
  using U_v     = Morpheus::UnaryContainer<con, v>;
  using ref_U_v = Morpheus::DenseVector<f>;
  VALIDATE_UNARY_CONTAINER(res, U_v, ref_U_v, f, int, def_layout, def_space);

  using vi       = Morpheus::Set<f, i, D, D>;
  using U_vi     = Morpheus::UnaryContainer<con, vi>;
  using ref_U_vi = Morpheus::DenseVector<f, i>;
  VALIDATE_UNARY_CONTAINER(res, U_vi, ref_U_vi, f, i, def_layout, def_space);

  using vl       = Morpheus::Set<f, D, l, D>;
  using U_vl     = Morpheus::UnaryContainer<con, vl>;
  using ref_U_vl = Morpheus::DenseVector<f, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vl, ref_U_vl, f, int, l, def_space);

  using vs       = Morpheus::Set<f, D, D, s>;
  using U_vs     = Morpheus::UnaryContainer<con, vs>;
  using ref_U_vs = Morpheus::DenseVector<f, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vs, ref_U_vs, f, int,
                           typename s::array_layout, s);

  using vil       = Morpheus::Set<f, i, l, D>;
  using U_vil     = Morpheus::UnaryContainer<con, vil>;
  using ref_U_vil = Morpheus::DenseVector<f, i, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vil, ref_U_vil, f, i, l, def_space);

  using vis       = Morpheus::Set<f, i, D, s>;
  using U_vis     = Morpheus::UnaryContainer<con, vis>;
  using ref_U_vis = Morpheus::DenseVector<f, i, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vis, ref_U_vis, f, i, def_layout, s);

  using vls       = Morpheus::Set<f, D, l, s>;
  using U_vls     = Morpheus::UnaryContainer<con, vls>;
  using ref_U_vls = Morpheus::DenseVector<f, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vls, ref_U_vls, f, int, l, s);

  using vils       = Morpheus::Set<f, i, l, s>;
  using U_vils     = Morpheus::UnaryContainer<con, vils>;
  using ref_U_vils = Morpheus::DenseVector<f, i, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vils, ref_U_vils, f, i, l, s);
}

/**
 * @brief Checks the compile-time type generation of a \p CooMatrix container
 * from a set of template arguments.
 *
 */
TEST(ContainerFactoryTest, UnaryContainer_CooMatrix_FromSet) {
  using D   = Morpheus::Default;
  using f   = float;
  using i   = long long;
  using l   = Kokkos::LayoutLeft;
  using s   = Kokkos::Serial;
  using con = Morpheus::CooMatrix<double>;

  using def_space  = Kokkos::DefaultExecutionSpace;
  using def_layout = Kokkos::LayoutRight;
  bool res;

  using v       = Morpheus::Set<f, D, D, D>;
  using U_v     = Morpheus::UnaryContainer<con, v>;
  using ref_U_v = Morpheus::CooMatrix<f>;
  VALIDATE_UNARY_CONTAINER(res, U_v, ref_U_v, f, int, def_layout, def_space);

  using vi       = Morpheus::Set<f, i, D, D>;
  using U_vi     = Morpheus::UnaryContainer<con, vi>;
  using ref_U_vi = Morpheus::CooMatrix<f, i>;
  VALIDATE_UNARY_CONTAINER(res, U_vi, ref_U_vi, f, i, def_layout, def_space);

  using vl       = Morpheus::Set<f, D, l, D>;
  using U_vl     = Morpheus::UnaryContainer<con, vl>;
  using ref_U_vl = Morpheus::CooMatrix<f, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vl, ref_U_vl, f, int, l, def_space);

  using vs       = Morpheus::Set<f, D, D, s>;
  using U_vs     = Morpheus::UnaryContainer<con, vs>;
  using ref_U_vs = Morpheus::CooMatrix<f, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vs, ref_U_vs, f, int,
                           typename s::array_layout, s);

  using vil       = Morpheus::Set<f, i, l, D>;
  using U_vil     = Morpheus::UnaryContainer<con, vil>;
  using ref_U_vil = Morpheus::CooMatrix<f, i, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vil, ref_U_vil, f, i, l, def_space);

  using vis       = Morpheus::Set<f, i, D, s>;
  using U_vis     = Morpheus::UnaryContainer<con, vis>;
  using ref_U_vis = Morpheus::CooMatrix<f, i, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vis, ref_U_vis, f, i, def_layout, s);

  using vls       = Morpheus::Set<f, D, l, s>;
  using U_vls     = Morpheus::UnaryContainer<con, vls>;
  using ref_U_vls = Morpheus::CooMatrix<f, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vls, ref_U_vls, f, int, l, s);

  using vils       = Morpheus::Set<f, i, l, s>;
  using U_vils     = Morpheus::UnaryContainer<con, vils>;
  using ref_U_vils = Morpheus::CooMatrix<f, i, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vils, ref_U_vils, f, i, l, s);
}

/**
 * @brief Checks the compile-time type generation of a \p DynamicVector
 * container from a set of template arguments.
 *
 */
TEST(ContainerFactoryTest, UnaryContainer_DynamicMatrix_FromSet) {
  using D   = Morpheus::Default;
  using f   = float;
  using i   = long long;
  using l   = Kokkos::LayoutLeft;
  using s   = Kokkos::Serial;
  using con = Morpheus::DynamicMatrix<double>;

  using def_space  = Kokkos::DefaultExecutionSpace;
  using def_layout = Kokkos::LayoutRight;
  bool res;

  using v       = Morpheus::Set<f, D, D, D>;
  using U_v     = Morpheus::UnaryContainer<con, v>;
  using ref_U_v = Morpheus::DynamicMatrix<f>;
  VALIDATE_UNARY_CONTAINER(res, U_v, ref_U_v, f, int, def_layout, def_space);

  using vi       = Morpheus::Set<f, i, D, D>;
  using U_vi     = Morpheus::UnaryContainer<con, vi>;
  using ref_U_vi = Morpheus::DynamicMatrix<f, i>;
  VALIDATE_UNARY_CONTAINER(res, U_vi, ref_U_vi, f, i, def_layout, def_space);

  using vl       = Morpheus::Set<f, D, l, D>;
  using U_vl     = Morpheus::UnaryContainer<con, vl>;
  using ref_U_vl = Morpheus::DynamicMatrix<f, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vl, ref_U_vl, f, int, l, def_space);

  using vs       = Morpheus::Set<f, D, D, s>;
  using U_vs     = Morpheus::UnaryContainer<con, vs>;
  using ref_U_vs = Morpheus::DynamicMatrix<f, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vs, ref_U_vs, f, int,
                           typename s::array_layout, s);

  using vil       = Morpheus::Set<f, i, l, D>;
  using U_vil     = Morpheus::UnaryContainer<con, vil>;
  using ref_U_vil = Morpheus::DynamicMatrix<f, i, l>;
  VALIDATE_UNARY_CONTAINER(res, U_vil, ref_U_vil, f, i, l, def_space);

  using vis       = Morpheus::Set<f, i, D, s>;
  using U_vis     = Morpheus::UnaryContainer<con, vis>;
  using ref_U_vis = Morpheus::DynamicMatrix<f, i, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vis, ref_U_vis, f, i, def_layout, s);

  using vls       = Morpheus::Set<f, D, l, s>;
  using U_vls     = Morpheus::UnaryContainer<con, vls>;
  using ref_U_vls = Morpheus::DynamicMatrix<f, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vls, ref_U_vls, f, int, l, s);

  using vils       = Morpheus::Set<f, i, l, s>;
  using U_vils     = Morpheus::UnaryContainer<con, vils>;
  using ref_U_vils = Morpheus::DynamicMatrix<f, i, l, s>;
  VALIDATE_UNARY_CONTAINER(res, U_vils, ref_U_vils, f, i, l, s);
}

TEST(ContainerFactoryTest, SetToUnaryContainerTypeList) {
  // Types
  struct A {};
  struct B {};
  struct C {};
  struct D {};
  struct E {};
  struct F {};
  using value_list  = Morpheus::TypeList<A, B>;
  using index_list  = Morpheus::TypeList<C, D>;
  using layout_list = Morpheus::TypeList<E, F>;
  //   using space_list  = Morpheus::TypeList<
  // // #if defined(MORPHEUS_ENABLE_SERIAL)
  // //       Kokkos::Serial,
  // // #endif
  // // #if defined(MORPHEUS_ENABLE_OPENMP)
  // //       Kokkos::OpenMP,
  // // #endif
  // //       Morpheus::Default>;

  // using vxi   = typename Morpheus::cross_product<value_list,
  // index_list>::type; using vxixl = typename Morpheus::cross_product<vxi,
  // layout_list>::type; using vxixlxs   = typename
  // Morpheus::cross_product<vxixl, space_list>::type;
  // <A, B> x <C, D> x <E, F>
  // = <A, B> x <[C, E], [C, F], [D, E], [D, F]>
  // = <[A, C, E], [A, C, F], [A, D, E], [A, D, F]
  //    [B, C, E], [B, C, F], [B, D, E], [B, D, F]>
  using types_set = typename Morpheus::cross_product<
      value_list,
      typename Morpheus::cross_product<index_list, layout_list>::type>::type;
  using ref_set =
      Morpheus::TypeList<Morpheus::Set<A, C, E>, Morpheus::Set<A, C, F>,
                         Morpheus::Set<A, D, E>, Morpheus::Set<A, D, F>,
                         Morpheus::Set<B, C, E>, Morpheus::Set<B, C, F>,
                         Morpheus::Set<B, D, E>, Morpheus::Set<B, D, F>>;
  bool res = std::is_same<types_set, ref_set>::value;
  EXPECT_EQ(res, 1);
}

// TEST(ContainerFactoryTest, SetToUnaryContainerTypeList) {
//   using value_list  = Morpheus::TypeList<double, float, int>;
//   using index_list  = Morpheus::TypeList<int, long long,
// Morpheus::Default > ;
//   using layout_list = Morpheus::TypeList<Kokkos::LayoutRight,
//                                          Kokkos::LayoutLeft,
//                                          Morpheus::Default>;
//   using space_list  = Morpheus::TypeList<
// #if defined(MORPHEUS_ENABLE_SERIAL)
//       Kokkos::Serial,
// #endif
// #if defined(MORPHEUS_ENABLE_OPENMP)
//       Kokkos::OpenMP,
// #endif
//       Morpheus::Default>;

//   // using vxi   = typename Morpheus::cross_product<value_list,
//   // index_list>::type; using vxixl = typename
// Morpheus::cross_product < vxi,
//   // layout_list>::type; using vxixlxs   = typename
//   // Morpheus::cross_product<vxixl, space_list>::type;
//   using types_set = typename Morpheus::cross_product<
//       value_list,
//       typename Morpheus::cross_product<
//           index_list, typename Morpheus::cross_product<
//                           layout_list, space_list>::type>::type>::type;
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_METAPROGRAMMING_HPP
