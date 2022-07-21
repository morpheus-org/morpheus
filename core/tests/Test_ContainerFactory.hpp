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

template <typename... Ts>
struct to_gtest_types {};

template <typename... Ts>
struct to_gtest_types<Morpheus::TypeList<Ts...>> {
  using type = ::testing::Types<Ts...>;
};

template <typename... Ts>
using vec = Morpheus::DenseVector<Ts...>;

TEST(ContainerFactoryTest, GenerateUnaryTypeList) {
  using D  = Morpheus::Default;
  using d  = double;
  using f  = float;
  using i  = int;
  using ll = long long;
  using l  = Kokkos::LayoutLeft;
  using r  = Kokkos::LayoutRight;
  using s  = TEST_EXECSPACE;

  using value_tlist  = Morpheus::TypeList<d, f, i>;
  using index_tlist  = Morpheus::TypeList<i, ll, D>;
  using layout_tlist = Morpheus::TypeList<r, l, D>;
  using space_tlist  = Morpheus::TypeList<s, D>;

  using types_set = typename Morpheus::cross_product<
      value_tlist,
      typename Morpheus::cross_product<
          index_tlist, typename Morpheus::cross_product<
                           layout_tlist, space_tlist>::type>::type>::type;
  //  Convert types set into container set
  using unary_types =
      typename Morpheus::generate_unary_typelist<Morpheus::DenseVector<double>,
                                                 types_set>::type;

  using res_t = Morpheus::TypeList<
      vec<d, i, r, s>, vec<d, i, r>, vec<d, i, l, s>, vec<d, i, l>,
      vec<d, i, s>, vec<d, i>, vec<d, ll, r, s>, vec<d, ll, r>,
      vec<d, ll, l, s>, vec<d, ll, l>, vec<d, ll, s>, vec<d, ll>, vec<d, r, s>,
      vec<d, r>, vec<d, l, s>, vec<d, l>, vec<d, s>, vec<d>, vec<f, i, r, s>,
      vec<f, i, r>, vec<f, i, l, s>, vec<f, i, l>, vec<f, i, s>, vec<f, i>,
      vec<f, ll, r, s>, vec<f, ll, r>, vec<f, ll, l, s>, vec<f, ll, l>,
      vec<f, ll, s>, vec<f, ll>, vec<f, r, s>, vec<f, r>, vec<f, l, s>,
      vec<f, l>, vec<f, s>, vec<f>, vec<i, i, r, s>, vec<i, i, r>,
      vec<i, i, l, s>, vec<i, i, l>, vec<i, i, s>, vec<i, i>, vec<i, ll, r, s>,
      vec<i, ll, r>, vec<i, ll, l, s>, vec<i, ll, l>, vec<i, ll, s>, vec<i, ll>,
      vec<i, r, s>, vec<i, r>, vec<i, l, s>, vec<i, l>, vec<i, s>, vec<i>>;

  bool res = std::is_same<unary_types, res_t>::value;
  EXPECT_EQ(res, 1);
  //  Convert to Types
  using DenseVectorTypes = typename to_gtest_types<unary_types>::type;
}

}  // namespace Test

#endif  // TEST_CORE_TEST_METAPROGRAMMING_HPP
