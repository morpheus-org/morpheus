/**
 * Test_TypeTraits.hpp
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

#ifndef TEST_CORE_TEST_TYPETRAITS_HPP
#define TEST_CORE_TEST_TYPETRAITS_HPP

#include <Morpheus_Core.hpp>

#include <setup/ContainerDefinition_Utils.hpp>

namespace Test {

TEST(TypeTraitsTest, IsVariantBuiltInTypes) {
  using variant = std::variant<int, double, float>;

  bool res = Morpheus::is_variant_member_v<double, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<long long, variant>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsVariantMorpheusTypes) {
  using CooDouble = Morpheus::CooMatrix<double>;
  using CsrDouble = Morpheus::CsrMatrix<double>;
  using variant   = std::variant<CooDouble, CsrDouble>;

  bool res = Morpheus::is_variant_member_v<CooDouble, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<Morpheus::CooMatrix<double, long long>,
                                      variant>;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_variant_member_v<Morpheus::CooMatrix<double, int>, variant>;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_variant_member_v<typename Morpheus::CooMatrix<double>::type,
                                    variant>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsVariantMorpheusTypesDefault) {
  using Coo     = typename Morpheus::CooMatrix<double>::type;
  using Csr     = typename Morpheus::CsrMatrix<double>::type;
  using variant = std::variant<Coo, Csr>;

  bool res = Morpheus::is_variant_member_v<
      Morpheus::CooMatrix<
          double, int, typename Kokkos::DefaultExecutionSpace::array_layout,
          Kokkos::DefaultExecutionSpace, typename Kokkos::MemoryManaged>,
      variant>;
  EXPECT_EQ(res, 1);

  // res = Morpheus::is_variant_member< Morpheus::CooMatrix<double, long long>,
  // variant>::value; EXPECT_EQ(res, 0);

  // res = Morpheus::is_variant_member< Morpheus::CooMatrix<double, int>,
  // variant>::value; EXPECT_EQ(res, 0);

  // res = Morpheus::is_variant_member<typename
  // Morpheus::CooMatrix<double>::type, variant>::value; EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, MemberTag) {
  bool res = Morpheus::has_tag_trait<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_tag_trait<int>::value;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsMatrixContainer) {
  bool res = Morpheus::is_matrix_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_matrix_container<Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_matrix_container<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container<Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_matrix_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_matrix_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsSparseMatrixContainer) {
  bool res =
      Morpheus::is_sparse_matrix_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_sparse_matrix_container<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container<
      Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container<
      Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_sparse_matrix_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsDenseMatrixContainer) {
  bool res =
      Morpheus::is_dense_matrix_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container<int>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_dense_matrix_container<Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_dense_matrix_container<Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_dense_matrix_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsVectorContainer) {
  bool res = Morpheus::is_vector_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container<Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container<Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_vector_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 1);
}

TEST(TypeTraitsTest, IsContainer) {
  bool res = Morpheus::is_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container<Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_container<Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 1);
}

TEST(TypeTraitsTest, IsDynamicContainer) {
  bool res =
      Morpheus::is_dynamic_matrix_container<Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<
      Morpheus::DenseMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<
      Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<
      Morpheus::DynamicMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_dynamic_matrix_container_v<Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container_v<Morpheus::DenseMatrix<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container_v<Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_dynamic_matrix_container_v<Morpheus::DynamicMatrix<double>>;
  EXPECT_EQ(res, 1);
}

TEST(TypeTraitsTest, IsSameFormat) {
  // Same types
  bool res = Morpheus::is_same_format<Morpheus::CooMatrix<double>,
                                      Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format<Morpheus::CooMatrix<double>,
                                 Morpheus::CooMatrix<float>>::value;
  EXPECT_EQ(res, 1);

  // Same Matrix Format but different type
  res = Morpheus::is_same_format<Morpheus::CooMatrix<double>,
                                 Morpheus::CooMatrix<double, long long>>::value;
  EXPECT_EQ(res, 1);

  // DynamicMatrices
  res = Morpheus::is_same_format<Morpheus::DynamicMatrix<double>,
                                 Morpheus::DynamicMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format<Morpheus::DynamicMatrix<double>,
                                 Morpheus::DynamicMatrix<float>>::value;
  EXPECT_EQ(res, 1);

  // Different Matrix Formats
  res = Morpheus::is_same_format<Morpheus::CooMatrix<double>,
                                 Morpheus::CsrMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  // Dense and Sparse Container
  res = Morpheus::is_same_format<Morpheus::DenseMatrix<double>,
                                 Morpheus::CsrMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  // Dense and Dense Container
  res = Morpheus::is_same_format<Morpheus::DenseMatrix<double>,
                                 Morpheus::DenseVector<double>>::value;
  EXPECT_EQ(res, 0);

  // Morpheus Container with built-in type
  res = Morpheus::is_same_format<Morpheus::DenseMatrix<double>, int>::value;
  EXPECT_EQ(res, 0);

  // Built-in types
  res = Morpheus::is_same_format_v<int, int>;
  EXPECT_EQ(res, 0);

  // Same types
  res = Morpheus::is_same_format_v<Morpheus::CooMatrix<double>,
                                   Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format_v<Morpheus::CooMatrix<double>,
                                   Morpheus::CooMatrix<float>>;
  EXPECT_EQ(res, 1);

  // Same Matrix Format but different type
  res = Morpheus::is_same_format_v<Morpheus::CooMatrix<double>,
                                   Morpheus::CooMatrix<double, long long>>;
  EXPECT_EQ(res, 1);

  // DynamicMatrices
  res = Morpheus::is_same_format_v<Morpheus::DynamicMatrix<double>,
                                   Morpheus::DynamicMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format_v<Morpheus::DynamicMatrix<double>,
                                   Morpheus::DynamicMatrix<float>>;
  EXPECT_EQ(res, 1);

  // Different Matrix Formats
  res = Morpheus::is_same_format_v<Morpheus::CooMatrix<double>,
                                   Morpheus::CsrMatrix<double>>;
  EXPECT_EQ(res, 0);

  // Dense and Sparse Container
  res = Morpheus::is_same_format_v<Morpheus::DenseMatrix<double>,
                                   Morpheus::CsrMatrix<double>>;
  EXPECT_EQ(res, 0);

  // Dense and Dense Container
  res = Morpheus::is_same_format_v<Morpheus::DenseMatrix<double>,
                                   Morpheus::DenseVector<double>>;
  EXPECT_EQ(res, 0);

  // Morpheus Container with built-in type
  res = Morpheus::is_same_format_v<Morpheus::DenseMatrix<double>, int>;
  EXPECT_EQ(res, 0);

  // Built-in types
  res = Morpheus::is_same_format_v<int, int>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, InSameMemorySpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, HaveSameLayout) {
  // Explicit Kokkos Layouts check
  bool res =
      Morpheus::have_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout<Kokkos::LayoutRight,
                                   Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout<Kokkos::LayoutLeft,
                                   Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 0);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout<Kokkos::LayoutRight,
                                   Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_layout<int, int>::value;
  EXPECT_EQ(res, 0);

  // Same containers with same default layout
  res = Morpheus::have_same_layout<Morpheus::CooMatrix<double>,
                                   Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  // Same containers with explicit same layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>>::value;
  EXPECT_EQ(res, 1);

  // Same containers with explicit same layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>,
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  // Same containers with explicit different layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Different containers with same default layout
  res = Morpheus::have_same_layout<Morpheus::CooMatrix<double>,
                                   Morpheus::CsrMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  // Different containers with explicit same layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutLeft>>::value;
  EXPECT_EQ(res, 1);

  // Different containers with explicit same layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  // Different containers with explicit different layout
  res = Morpheus::have_same_layout<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout_v<Kokkos::LayoutLeft, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout_v<Kokkos::LayoutRight, Kokkos::LayoutRight>;
  EXPECT_EQ(res, 1);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout_v<Kokkos::LayoutLeft, Kokkos::LayoutRight>;
  EXPECT_EQ(res, 0);

  // Explicit Kokkos Layouts check
  res = Morpheus::have_same_layout_v<Kokkos::LayoutRight, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_layout_v<int, int>;
  EXPECT_EQ(res, 0);

  // Same containers with same default layout
  res = Morpheus::have_same_layout_v<Morpheus::CooMatrix<double>,
                                     Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  // Same containers with explicit same layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>>;
  EXPECT_EQ(res, 1);

  // Same containers with explicit same layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>,
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);

  // Same containers with explicit different layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);

  // Different containers with same default layout
  res = Morpheus::have_same_layout_v<Morpheus::CooMatrix<double>,
                                     Morpheus::CsrMatrix<double>>;
  EXPECT_EQ(res, 1);

  // Different containers with explicit same layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutLeft>>;
  EXPECT_EQ(res, 1);

  // Different containers with explicit same layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutRight>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);

  // Different containers with explicit different layout
  res = Morpheus::have_same_layout_v<
      Morpheus::CooMatrix<double, Kokkos::LayoutLeft>,
      Morpheus::CsrMatrix<double, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, HaveSameValueType) {
  // Built-in types
  bool res = Morpheus::have_same_value_type<int, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<float, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<int, float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<float, float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<Morpheus::ValueType<int>,
                                       Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::ValueType<float>,
                                       Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<Morpheus::ValueType<int>,
                                       Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<Morpheus::ValueType<float>,
                                       Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 1);

  // Same container with same value type
  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<double>,
                                       Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<float>,
                                       Morpheus::CooMatrix<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<int>,
                                       Morpheus::CooMatrix<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<long long>,
                                       Morpheus::CooMatrix<long long>>::value;
  EXPECT_EQ(res, 1);

  // Same container with different value type
  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<double>,
                                       Morpheus::CooMatrix<float>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<float>,
                                       Morpheus::CooMatrix<int>>::value;
  EXPECT_EQ(res, 0);

  // Different container with same value type
  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<double>,
                                       Morpheus::CsrMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CsrMatrix<float>,
                                       Morpheus::CooMatrix<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<int>,
                                       Morpheus::CsrMatrix<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type<Morpheus::CsrMatrix<long long>,
                                       Morpheus::CooMatrix<long long>>::value;
  EXPECT_EQ(res, 1);

  // Different container with different value type
  res = Morpheus::have_same_value_type<Morpheus::CooMatrix<double>,
                                       Morpheus::CsrMatrix<float>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type<Morpheus::CsrMatrix<float>,
                                       Morpheus::CooMatrix<int>>::value;
  EXPECT_EQ(res, 0);
  ////////////////////////
  res = Morpheus::have_same_value_type_v<int, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<float, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<int, float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<float, float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<Morpheus::ValueType<int>,
                                         Morpheus::ValueType<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::ValueType<float>,
                                         Morpheus::ValueType<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<Morpheus::ValueType<int>,
                                         Morpheus::ValueType<float>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<Morpheus::ValueType<float>,
                                         Morpheus::ValueType<float>>;
  EXPECT_EQ(res, 1);

  // Same container with same value type
  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<double>,
                                         Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<float>,
                                         Morpheus::CooMatrix<float>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<int>,
                                         Morpheus::CooMatrix<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<long long>,
                                         Morpheus::CooMatrix<long long>>;
  EXPECT_EQ(res, 1);

  // Same container with different value type
  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<double>,
                                         Morpheus::CooMatrix<float>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<float>,
                                         Morpheus::CooMatrix<int>>;
  EXPECT_EQ(res, 0);

  // Different container with same value type
  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<double>,
                                         Morpheus::CsrMatrix<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CsrMatrix<float>,
                                         Morpheus::CooMatrix<float>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<int>,
                                         Morpheus::CsrMatrix<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::have_same_value_type_v<Morpheus::CsrMatrix<long long>,
                                         Morpheus::CooMatrix<long long>>;
  EXPECT_EQ(res, 1);

  // Different container with different value type
  res = Morpheus::have_same_value_type_v<Morpheus::CooMatrix<double>,
                                         Morpheus::CsrMatrix<float>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::have_same_value_type_v<Morpheus::CsrMatrix<float>,
                                         Morpheus::CooMatrix<int>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, HaveSameIndexType) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsCompatible) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsDynamicallyCompatible) {
  // Same types but not dynamic containers
  bool res =
      Morpheus::is_dynamically_compatible<Morpheus::CooMatrix<double>,
                                          Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 0);

  // Same template parameters with Dynamic container as T2
  res = Morpheus::is_dynamically_compatible<
      Morpheus::CooMatrix<double>, Morpheus::DynamicMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  // Same template parameters with Dynamic container as T1
  res = Morpheus::is_dynamically_compatible<Morpheus::DynamicMatrix<double>,
                                            Morpheus::CooMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  // Both dynamic containers with the same template parameters
  res = Morpheus::is_dynamically_compatible<
      Morpheus::DynamicMatrix<double>, Morpheus::DynamicMatrix<double>>::value;
  EXPECT_EQ(res, 1);

  // Both dynamic containers with different template parameters
  res = Morpheus::is_dynamically_compatible<
      Morpheus::DynamicMatrix<double>, Morpheus::DynamicMatrix<float>>::value;
  EXPECT_EQ(res, 0);

  // Different template parameters with dynamic container as T1
  res = Morpheus::is_dynamically_compatible<Morpheus::DynamicMatrix<double>,
                                            Morpheus::CooMatrix<float>>::value;
  EXPECT_EQ(res, 0);

  // Different template parameters with dynamic container as T2
  res = Morpheus::is_dynamically_compatible<
      Morpheus::CooMatrix<double>, Morpheus::DynamicMatrix<float>>::value;
  EXPECT_EQ(res, 0);

  // Same types but not dynamic containers
  res = Morpheus::is_dynamically_compatible_v<Morpheus::CooMatrix<double>,
                                              Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 0);

  // Same template parameters with Dynamic container as T2
  res = Morpheus::is_dynamically_compatible_v<Morpheus::CooMatrix<double>,
                                              Morpheus::DynamicMatrix<double>>;
  EXPECT_EQ(res, 1);

  // Same template parameters with Dynamic container as T1
  res = Morpheus::is_dynamically_compatible_v<Morpheus::DynamicMatrix<double>,
                                              Morpheus::CooMatrix<double>>;
  EXPECT_EQ(res, 1);

  // Both dynamic containers with the same template parameters
  res = Morpheus::is_dynamically_compatible_v<Morpheus::DynamicMatrix<double>,
                                              Morpheus::DynamicMatrix<double>>;
  EXPECT_EQ(res, 1);

  // Both dynamic containers with different template parameters
  res = Morpheus::is_dynamically_compatible_v<Morpheus::DynamicMatrix<double>,
                                              Morpheus::DynamicMatrix<float>>;
  EXPECT_EQ(res, 0);

  // Different template parameters with dynamic container as T1
  res = Morpheus::is_dynamically_compatible_v<Morpheus::DynamicMatrix<double>,
                                              Morpheus::CooMatrix<float>>;
  EXPECT_EQ(res, 0);

  // Different template parameters with dynamic container as T2
  res = Morpheus::is_dynamically_compatible_v<Morpheus::CooMatrix<double>,
                                              Morpheus::DynamicMatrix<float>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsCompatibleType) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsCompatibleTypeFromDifferentSpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, RemoveCVRef) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsArithmetic) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsExecutionSpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsHostMemorySpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsHostSpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IsSerialSpace) { EXPECT_EQ(0, 1); }

#if defined(MORPHEUS_ENABLE_OPENMP)
TEST(TypeTraitsTest, IsOpenMPSpace) { EXPECT_EQ(0, 1); }
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
TEST(TypeTraitsTest, IsCudaSpace) { EXPECT_EQ(0, 1); }
#endif

TEST(TypeTraitsTest, HasAccess) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, HasKokkosSpace) { EXPECT_EQ(0, 1); }

TEST(TypeTraitsTest, IssKokkosSpace) { EXPECT_EQ(0, 1); }
}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
