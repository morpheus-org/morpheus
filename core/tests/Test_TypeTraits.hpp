/**
 * Test_TypeTraits.hpp
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

#ifndef TEST_CORE_TEST_TYPETRAITS_HPP
#define TEST_CORE_TEST_TYPETRAITS_HPP

#include <Morpheus_Core.hpp>

namespace Impl {
template <typename T>
struct with_tag {
  using tag = T;
};

struct no_traits {};
}  // namespace Impl

namespace Test {

/**
 * @brief The \p is_variant_member checks if the passed type is a member of
 * the variant container
 *
 */
TEST(TypeTraitsTest, IsVariantMember) {
  using variant = Morpheus::Impl::Variant::variant<int, double, float>;

  bool res = Morpheus::is_variant_member<double, variant>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member<float, variant>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<int, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member<long long, variant>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_variant_member_v<char, variant>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_variant_member checks if the passed type is a member of
 * the variant container
 *
 */
TEST(TypeTraitsTest, IsVariantMemberStruct) {
  struct A {
    using type = A;
  };

  struct B {
    using type = B;
  };

  struct C {
    using type = C;
  };

  using variant = Morpheus::Impl::Variant::variant<A, B>;

  bool res = Morpheus::is_variant_member<A, variant>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<B, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member<C, variant>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_variant_member_v<C, variant>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_tag_trait checks if the passed type has the \p tag
 member
 * trait so we check custom types for that
 *
 */
TEST(TypeTraitsTest, MemberTag) {
  bool res = Morpheus::has_tag_trait<Impl::with_tag<void>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_tag_trait<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_tag_trait<int>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_tag_trait_v<Impl::with_tag<void>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_tag_trait_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_layout checks if the passed type is a valid layout. For
 * the check to be valid, the type must be one of the supported layouts.
 *
 */
TEST(TypeTraitsTest, IsLayout) {
  bool res = Morpheus::is_layout<Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_layout<typename Kokkos::LayoutLeft::array_layout>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_layout<Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  // Valid Layout but Not Supported
  res = Morpheus::is_layout<Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_layout<int>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_layout_v<Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_layout_v<int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_layout checks if the passed type has a valid layout. For
 * the check to be valid, the type must be a valid layout and have an
 * \p array_layout trait.
 *
 */
TEST(TypeTraitsTest, HasLayout) {
  bool res = Morpheus::has_layout<Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_layout<Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  // Has a layout but is also a layout itself
  res = Morpheus::has_layout<Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  // Has Layout but Not Supported
  res = Morpheus::has_layout<Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::has_layout<int>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_layout_v<Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::has_layout_v<int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_same_layout checks if the two types passed are the
 * same layout. For the check to be valid, both types must be a valid
 * layout and be the same.
 *
 */
TEST(TypeTraitsTest, IsSameLayout) {
  bool res =
      Morpheus::is_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_layout<
      typename Kokkos::LayoutLeft::array_layout,
      typename Kokkos::LayoutLeft::array_layout>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 0);

  // Same Layout but Not Supported
  res = Morpheus::is_same_layout<Kokkos::LayoutStride,
                                 Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_same_layout<int, int>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_same_layout_v<Kokkos::LayoutLeft, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_same_layout_v<int, int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_same_layout checks if the two types passed have the
 * same layout. For the check to be valid, both types must have an
 * \p array_layout trait and the \p is_same_layout must be satisfied.
 *
 */
TEST(TypeTraitsTest, HasSameLayout) {
  bool res =
      Morpheus::has_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  // Both valid layouts but also have layout trait
  res = Morpheus::has_same_layout<
      typename Kokkos::LayoutLeft::array_layout,
      typename Kokkos::LayoutLeft::array_layout>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::has_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 0);

  struct A {
    using array_layout = Kokkos::LayoutLeft;
  };

  struct B {
    using array_layout = Kokkos::LayoutRight;
  };

  struct C {
    using array_layout = Kokkos::LayoutLeft;
  };

  res = Morpheus::has_same_layout<A, B>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_layout<A, C>::value;
  EXPECT_EQ(res, 1);

  // Same Layout but Not Supported
  res = Morpheus::has_same_layout<Kokkos::LayoutStride,
                                  Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::has_same_layout<int, int>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_same_layout_v<Kokkos::LayoutLeft, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::has_same_layout_v<int, int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_value_type checks if the passed type is a valid value
 * type. For the check to be valid, the type must be a scalar.
 *
 */
TEST(TypeTraitsTest, IsValueType) {
  bool res = Morpheus::is_value_type<
      typename Morpheus::ValueType<float>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<
      typename Morpheus::ValueType<double>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<
      typename Morpheus::ValueType<int>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<
      typename Morpheus::ValueType<long long>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<float>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<int>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<std::vector<double>>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_value_type<typename std::vector<double>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<std::vector<std::string>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_value_type_v<float>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<std::vector<double>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_value_type checks if the passed type has a valid value
 * type. For the check to be valid, the type must be a valid value type and
 * have a \p value_type trait.
 *
 */
TEST(TypeTraitsTest, HasValueType) {
  bool res = Morpheus::has_value_type<Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_value_type<Morpheus::ValueType<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_value_type<Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_value_type<Morpheus::ValueType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_value_type<float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_value_type<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_value_type<std::vector<double>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::has_value_type<typename std::vector<double>::value_type>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_value_type<std::vector<std::string>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_value_type_v<std::vector<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_value_type_v<typename std::vector<double>::value_type>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_same_value_type checks if the two types passed are the
 * same value type. For the check to be valid, both types must be a valid
 * value type and be the same.
 *
 */
TEST(TypeTraitsTest, IsSameValueType) {
  bool res = Morpheus::is_same_value_type<
      typename Morpheus::ValueType<float>::value_type,
      typename Morpheus::ValueType<float>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<
      typename Morpheus::ValueType<float>::value_type,
      typename Morpheus::ValueType<double>::value_type>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<
      typename Morpheus::ValueType<int>::value_type,
      typename Morpheus::ValueType<int>::value_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<
      typename Morpheus::ValueType<int>::value_type, int>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<double, double>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<long long, long long>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<double, long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<std::vector<double>,
                                     std::vector<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<
      typename std::vector<double>::value_type,
      typename std::vector<double>::value_type>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::is_same_value_type_v<float, float>;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_value_type_v<std::vector<double>, std::vector<double>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_same_value_type checks if the two types passed have the
 * same value type. For the check to be valid, both types must have a
 * \p value_type trait and the \p is_same_value_type must be satisfied.
 *
 */
TEST(TypeTraitsTest, HasSameValueType) {
  bool res = Morpheus::has_same_value_type<Morpheus::ValueType<float>,
                                           Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_value_type<Morpheus::ValueType<float>,
                                      Morpheus::ValueType<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_value_type<Morpheus::ValueType<int>,
                                      Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_value_type<Morpheus::ValueType<int>, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_value_type<float, float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_value_type<int, float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_value_type<std::vector<double>,
                                      std::vector<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_value_type<std::vector<double>, double>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res =
      Morpheus::has_same_value_type_v<std::vector<double>, std::vector<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_value_type_v<std::vector<double>, double>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_index_type checks if the passed type is a valid index
 * type. For the check to be valid, the type must be an integral.
 *
 */
TEST(TypeTraitsTest, IsIndexType) {
  bool res = Morpheus::is_index_type<
      typename Morpheus::IndexType<int>::index_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type<
      typename Morpheus::IndexType<long long>::index_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type<float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type<int>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type<std::vector<int>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_index_type_v<float>;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_index_type_v<typename Morpheus::IndexType<int>::index_type>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p has_index_type checks if the passed type has a valid index
 * type. For the check to be valid, the type must be a valid index type and
 * have a \p index_type trait.
 *
 */
TEST(TypeTraitsTest, HasIndexType) {
  bool res = Morpheus::has_index_type<
      typename Morpheus::IndexType<int>::index_type>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_index_type<Morpheus::IndexType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_index_type<Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_index_type<float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_index_type<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_index_type<std::vector<int>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_index_type_v<float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_index_type_v<Morpheus::IndexType<int>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_same_index_type checks if the two types passed are the
 * same index type. For the check to be valid, both types must be a valid
 * index type and be the same.
 *
 */
TEST(TypeTraitsTest, IsSameIndexType) {
  bool res = Morpheus::is_same_index_type<
      typename Morpheus::IndexType<int>::index_type,
      typename Morpheus::IndexType<int>::index_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<
      typename Morpheus::IndexType<int>::index_type,
      typename Morpheus::IndexType<long long>::index_type>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<
      typename Morpheus::IndexType<long long>::index_type,
      typename Morpheus::IndexType<long long>::index_type>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<
      typename Morpheus::IndexType<int>::index_type, int>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<
      typename Morpheus::IndexType<int>::index_type, long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<int, double>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<long long, long long>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<int, int>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<int, long long>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_same_index_type_v<
      typename Morpheus::IndexType<int>::index_type, int>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type_v<int, long long>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_same_index_type checks if the two types passed have the
 * same index type. For the check to be valid, both types must have a
 * \p index_type  trait and the \p is_same_index_type  must be satisfied.
 *
 */
TEST(TypeTraitsTest, HasSameIndexType) {
  bool res = Morpheus::has_same_index_type<Morpheus::IndexType<int>,
                                           Morpheus::IndexType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_index_type<Morpheus::IndexType<int>,
                                      Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<Morpheus::IndexType<long long>,
                                      Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_index_type<
      typename Morpheus::IndexType<int>::index_type, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<Morpheus::IndexType<int>, int>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::has_same_index_type<Morpheus::IndexType<int>, long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<int, double>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<long long, long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<int, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type<int, long long>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_same_index_type_v<Morpheus::IndexType<int>, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_index_type_v<Morpheus::IndexType<int>,
                                        Morpheus::IndexType<int>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p remove_cvref removes the topmost const- and
 * reference-qualifiers of the type passed
 *
 */
TEST(TypeTraitsTest, RemoveCVRef) {
  bool res = std::is_const<Morpheus::remove_cvref<int>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<int>::type>::value;
  EXPECT_EQ(res, 0);

  res = std::is_const<Morpheus::remove_cvref<const int>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<const int>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_same<Morpheus::remove_cvref<const int>::type, int>::value;
  EXPECT_EQ(res, 1);

  res = std::is_const<Morpheus::remove_cvref<const int&>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<const int&>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_same<Morpheus::remove_cvref<const int&>::type, int>::value;
  EXPECT_EQ(res, 1);

  res = std::is_const<Morpheus::remove_cvref<int&>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<int&>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_same<Morpheus::remove_cvref<int&>::type, int>::value;
  EXPECT_EQ(res, 1);

  res = std::is_const<Morpheus::remove_cvref<int*>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<int*>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_same<Morpheus::remove_cvref<int*>::type, int*>::value;
  EXPECT_EQ(res, 1);

  // Removing const from `const int *` does not modify the type, because the
  // pointer itself is not const.
  res = std::is_const<Morpheus::remove_cvref<const int*>::type>::value;
  EXPECT_EQ(res, 0);
  res = std::is_reference<Morpheus::remove_cvref<const int*>::type>::value;
  EXPECT_EQ(res, 0);
  res =
      std::is_same<Morpheus::remove_cvref<const int*>::type, const int*>::value;
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
