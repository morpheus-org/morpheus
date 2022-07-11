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
namespace Impl {
template <typename T>
struct with_tag {
  using tag = T;
};

struct no_traits {};
}  // namespace Impl

namespace Test {

TEST(TypeTraitsTest, IsVariantBuiltInTypes) {
  using variant = std::variant<int, double, float>;

  bool res = Morpheus::is_variant_member_v<double, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<long long, variant>;
  EXPECT_EQ(res, 0);
  EXPECT_EQ(0, 1);
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
  EXPECT_EQ(0, 1);
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
  EXPECT_EQ(0, 1);

  // res = Morpheus::is_variant_member< Morpheus::CooMatrix<double, long long>,
  // variant>::value; EXPECT_EQ(res, 0);

  // res = Morpheus::is_variant_member< Morpheus::CooMatrix<double, int>,
  // variant>::value; EXPECT_EQ(res, 0);

  // res = Morpheus::is_variant_member<typename
  // Morpheus::CooMatrix<double>::type, variant>::value; EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_tag_trait checks if the passed type has the \p tag member
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

  res = Morpheus::has_tag_trait_v<Impl::with_tag<void>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_tag_trait_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_tag_trait_v<int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_matrix_container checks if the passed type has the \p tag
 * member trait and if it is a valid Matrix Tag.
 *
 */
TEST(TypeTraitsTest, IsMatrixContainer) {
  bool res = Morpheus::is_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Matrix Tag is a valid tag for Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>;
  EXPECT_EQ(res, 1);

  // A Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Matrix Container
  res = Morpheus::is_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_sparse_matrix_container checks if the passed type has the
 * \p tag member trait and if it is a valid Sparse Matrix Tag.
 *
 */
TEST(TypeTraitsTest, IsSparseMatrixContainer) {
  bool res = Morpheus::is_sparse_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 1);

  // A Sparse Matrix Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Matrix Tag is not a valid tag for Sparse Matrix Container - A sparse
  // matrix is a matrix but a matrix is not necessarilly sparse
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Vector Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Sparse Matrix Container
  res =
      Morpheus::is_sparse_matrix_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 1);

  // A Sparse Matrix Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 1);

  // A Matrix Tag is not a valid tag for Sparse Matrix Container - A sparse
  // matrix is a matrix but a matrix is not necessarilly sparse
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Vector Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_dense_matrix_container checks if the passed type has the
 * \p tag member trait and if it is a valid Dense Matrix Tag.
 *
 */
TEST(TypeTraitsTest, IsDenseMatrixContainer) {
  bool res = Morpheus::is_dense_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Dense Matrix Container - A dense
  // matrix is a matrix but a matrix is not necessarilly dense
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Dense Matrix Container - A dense
  // matrix is a matrix but a matrix is not necessarilly dense
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_vector_container checks if the passed type has the
 * \p tag member trait and if it is a valid Vector Tag.
 *
 */
TEST(TypeTraitsTest, IsVectorContainer) {
  bool res = Morpheus::is_vector_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Vector Container
  res = Morpheus::is_vector_container<Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_vector_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Vector Container
  res =
      Morpheus::is_vector_container_v<Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 1);

  // A Vector Tag is a valid tag for Vector Container
  res =
      Morpheus::is_vector_container_v<Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_dense_vector_container checks if the passed type has the
 * \p tag member trait and if it is a valid Dense Vector Tag.
 *
 */
TEST(TypeTraitsTest, IsDesnseVectorContainer) {
  bool res = Morpheus::is_dense_vector_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_vector_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Vector Tag is a valid tag for Dense Vector Container - A dense vector
  // is a vector but a vector is not necessarilly dense
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Vector Tag is a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container<
      Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_dense_vector_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_vector_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Vector Tag is not a valid tag for Dense Vector Container - A dense vector
  // is a vector but a vector is not necessarilly dense
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 0);

  // A Vector Tag is a valid tag for Dense Vector Container
  res = Morpheus::is_dense_vector_container_v<
      Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_container checks if the passed type has the \p tag member
 * trait and if it is a valid Morpheus Container i.e either a Vector or Matrix.
 *
 */
TEST(TypeTraitsTest, IsContainer) {
  bool res = Morpheus::is_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container_v<Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 1);

  // A Vector Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container_v<Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_dynamic_matrix_container checks if the passed type has the
 * \p tag member trait and if it is a valid Dynamic Matrix Container
 *
 */
TEST(TypeTraitsTest, IsDynamicContainer) {
  bool res = Morpheus::is_dynamic_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Vector Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dynamic Tag is a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DynamicTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_dynamic_matrix_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container_v<Impl::no_traits>;
  EXPECT_EQ(res, 0);

  // A COO Tag is not a valid tag for Dynamic Matrix Container
  res =
      Morpheus::is_dynamic_matrix_container_v<Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container_v<
      Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dense Vector Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container_v<
      Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 0);

  // A Dynamic Tag is a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container_v<
      Impl::with_tag<Morpheus::DynamicTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_same_format checks if the two types have the same format i.e
 * are valid Morpheus Containers and have the same tag as member trait.
 *
 */
TEST(TypeTraitsTest, IsSameFormat) {
  // Same types
  bool res = Morpheus::is_same_format<Impl::with_tag<Morpheus::CooTag>,
                                      Impl::with_tag<Morpheus::CooTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format<Impl::with_tag<Morpheus::DynamicTag>,
                                 Impl::with_tag<Morpheus::DynamicTag>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_format<Impl::with_tag<Morpheus::DenseVectorTag>,
                               Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_format<Impl::with_tag<Morpheus::DenseMatrixTag>,
                               Impl::with_tag<Morpheus::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format<
      Impl::with_tag<Morpheus::Impl::SparseMatTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 1);

  // Check against derived formats
  res = Morpheus::is_same_format<
      Impl::with_tag<Morpheus::CooTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_format<
      Impl::with_tag<Morpheus::CooTag>,
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // Check against different formats
  res = Morpheus::is_same_format<Impl::with_tag<Morpheus::CooTag>,
                                 Impl::with_tag<Morpheus::CsrTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_format<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                 Impl::with_tag<Morpheus::CsrTag>>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_same_format<Impl::with_tag<Morpheus::DenseMatrixTag>,
                               Impl::with_tag<Morpheus::DenseVectorTag>>::value;
  EXPECT_EQ(res, 0);

  // Morpheus format against built-in type
  res = Morpheus::is_same_format<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                 Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  // Built-in types
  res =
      Morpheus::is_same_format<Impl::with_tag<int>, Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  // Same types
  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::CooTag>,
                                   Impl::with_tag<Morpheus::CooTag>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DynamicTag>,
                                   Impl::with_tag<Morpheus::DynamicTag>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DenseVectorTag>,
                                   Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                   Impl::with_tag<Morpheus::DenseMatrixTag>>;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_format_v<Impl::with_tag<Morpheus::Impl::SparseMatTag>,
                                 Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 1);

  // Check against derived formats
  res =
      Morpheus::is_same_format_v<Impl::with_tag<Morpheus::CooTag>,
                                 Impl::with_tag<Morpheus::Impl::SparseMatTag>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::CooTag>,
                                   Impl::with_tag<Morpheus::Impl::MatrixTag>>;
  EXPECT_EQ(res, 0);

  // Check against different formats
  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::CooTag>,
                                   Impl::with_tag<Morpheus::CsrTag>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                   Impl::with_tag<Morpheus::CsrTag>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                   Impl::with_tag<Morpheus::DenseVectorTag>>;
  EXPECT_EQ(res, 0);

  // Morpheus format against built-in type
  res = Morpheus::is_same_format_v<Impl::with_tag<Morpheus::DenseMatrixTag>,
                                   Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  // Built-in types
  res = Morpheus::is_same_format_v<Impl::with_tag<int>, Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);
}

// /**
//  * @brief The \p is_memory_space checks if the type is a memory space i.e has
//  as
//  * a \p memory_space member trait it self.
//  *
//  */
// TEST(TypeTraitsTest, IsMemorySpace) {
//   // A structure like this meets the requirements of a valid memory space i.e
//   // has a memory_space trait that is the same as it's name
//   struct ValidTestSpace {
//     using memory_space = ValidTestSpace;
//   };

//   struct InvalidTestSpace {
//     using memory_space = int;
//   };

//   bool res = Morpheus::is_memory_space<ValidTestSpace>::value;
//   EXPECT_EQ(res, 1);

//   // Kokkos Memory Space
//   res = Morpheus::is_memory_space<Kokkos::HostSpace>::value;
//   EXPECT_EQ(res, 1);

//   // Kokkos Execution Space
//   res = Morpheus::is_memory_space<TEST_EXECSPACE>::value;
//   EXPECT_EQ(res, 0);

//   res = Morpheus::is_memory_space<InvalidTestSpace>::value;
//   EXPECT_EQ(res, 0);

//   res = Morpheus::is_memory_space_v<ValidTestSpace>;
//   EXPECT_EQ(res, 1);

//   // Kokkos Memory Space
//   res = Morpheus::is_memory_space_v<Kokkos::HostSpace>;
//   EXPECT_EQ(res, 1);

//   res = Morpheus::is_memory_space_v<InvalidTestSpace>;
//   EXPECT_EQ(res, 0);
// }

/**
 * @brief The \p is_memory_space checks if the type has a valid and
 * supported memory space.
 *
 */
TEST(TypeTraitsTest, IsMemorySpace) {
  // A structure like this meets the requirements of a valid memory space i.e
  // has a memory_space trait that is the same as it's name BUT this is not
  // supported as a MemorySpace
  struct TestSpace {
    using memory_space = TestSpace;
  };

  bool res = Morpheus::is_memory_space<TestSpace>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_memory_space<int>::value;
  EXPECT_EQ(res, 0);

// Kokkos Memory Space
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_memory_space<Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_memory_space<Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);
#endif

  // Kokkos Execution Space
  res = Morpheus::is_memory_space<TEST_EXECSPACE>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_memory_space<TestSpace>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_memory_space_v<int>;
  EXPECT_EQ(res, 0);

// Kokkos Memory Space
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_memory_space_v<Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_memory_space_v<Kokkos::CudaSpace>;
  EXPECT_EQ(res, 1);
#endif

  // Kokkos Execution Space
  res = Morpheus::is_memory_space_v<TEST_EXECSPACE>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_same_memory_space checks if two types are in the same valid
 * and supported memory space.
 *
 */
TEST(TypeTraitsTest, InSameMemorySpace) {
  struct TestSpace {
    using memory_space = TestSpace;
  };

  bool res = Morpheus::is_same_memory_space<TestSpace, TestSpace>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_same_memory_space<int, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_memory_space<TestSpace, int>::value;
  EXPECT_EQ(res, 0);

// Kokkos Memory Space
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_same_memory_space<Kokkos::HostSpace,
                                       Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_memory_space<Kokkos::HostSpace, TestSpace>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_same_memory_space<Kokkos::HostSpace,
                                       Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_memory_space<Kokkos::CudaSpace,
                                       Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_memory_space<Kokkos::CudaSpace, TestSpace>::value;
  EXPECT_EQ(res, 0);
#endif

  res = Morpheus::is_same_memory_space_v<TestSpace, TestSpace>;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_same_memory_space_v<int, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_memory_space_v<TestSpace, int>;
  EXPECT_EQ(res, 0);

// Kokkos Memory Space
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_same_memory_space_v<Kokkos::HostSpace, Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_memory_space_v<Kokkos::HostSpace, TestSpace>;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_same_memory_space_v<Kokkos::HostSpace, Kokkos::CudaSpace>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_memory_space_v<Kokkos::CudaSpace, Kokkos::CudaSpace>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_memory_space_v<Kokkos::CudaSpace, TestSpace>;
  EXPECT_EQ(res, 0);
#endif
}

/**
 * @brief The \p is_layout checks if the type has a valid and
 * supported layout i.e an \p array_layout member trait and is either
 * \p Kokkos::LayoutLeft or \p Kokkos::LayoutRight.
 *
 */
TEST(TypeTraitsTest, IsLayout) {
  struct TestLayout {
    using array_layout = TestLayout;
  };
  // Has array_layout but not valid
  bool res = Morpheus::is_layout<TestLayout>::value;
  EXPECT_EQ(res, 0);
  // Valid Layouts
  res = Morpheus::is_layout<Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_layout<Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  // Valid Kokkos Layout but not supported by Morpheus
  res = Morpheus::is_layout<Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Has array_layout but not valid
  res = Morpheus::is_layout_v<TestLayout>;
  EXPECT_EQ(res, 0);
  // Valid Layouts
  res = Morpheus::is_layout_v<Kokkos::LayoutRight>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_layout_v<Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Valid Kokkos Layout but not supported by Morpheus
  res = Morpheus::is_layout_v<Kokkos::LayoutStride>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_same_layout checks if two types are have the same valid and
 * supported layout.
 *
 */
TEST(TypeTraitsTest, IsSameLayout) {
  struct TestLayout {
    using array_layout = TestLayout;
  };
  // Has array_layout but not valid
  bool res = Morpheus::is_same_layout<TestLayout, TestLayout>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_layout<TestLayout, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 0);

  // Valid Layouts and same
  res = Morpheus::is_same_layout<Kokkos::LayoutLeft, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 1);

  // Valid but different layouts
  res =
      Morpheus::is_same_layout<Kokkos::LayoutRight, Kokkos::LayoutLeft>::value;
  EXPECT_EQ(res, 0);

  // Valid Layouts and same
  res =
      Morpheus::is_same_layout<Kokkos::LayoutRight, Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 1);

  // Valid Layouts and different - first not supported
  res = Morpheus::is_same_layout<Kokkos::LayoutStride,
                                 Kokkos::LayoutRight>::value;
  EXPECT_EQ(res, 0);

  // Valid Layouts but not supported so fails
  res = Morpheus::is_same_layout<Kokkos::LayoutStride,
                                 Kokkos::LayoutStride>::value;
  EXPECT_EQ(res, 0);

  // Has array_layout but not valid
  res = Morpheus::is_same_layout_v<TestLayout, TestLayout>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_layout_v<TestLayout, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 0);

  // Valid Layouts and same
  res = Morpheus::is_same_layout_v<Kokkos::LayoutLeft, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 1);

  // Valid but different layouts
  res = Morpheus::is_same_layout_v<Kokkos::LayoutRight, Kokkos::LayoutLeft>;
  EXPECT_EQ(res, 0);

  // Valid Layouts and same
  res = Morpheus::is_same_layout_v<Kokkos::LayoutRight, Kokkos::LayoutRight>;
  EXPECT_EQ(res, 1);

  // Valid Layouts and different - first not supported
  res = Morpheus::is_same_layout_v<Kokkos::LayoutStride, Kokkos::LayoutRight>;
  EXPECT_EQ(res, 0);

  // Valid Layouts but not supported so fails
  res = Morpheus::is_same_layout_v<Kokkos::LayoutStride, Kokkos::LayoutStride>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_value_type checks if the type has a valid value type i.e a
 * \p value_type member trait that is scalar.
 *
 */
TEST(TypeTraitsTest, IsValueType) {
  bool res = Morpheus::is_value_type<Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<Morpheus::ValueType<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<Morpheus::ValueType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_value_type<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_value_type<std::vector<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type<std::vector<std::string>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_value_type_v<Morpheus::ValueType<float>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<Morpheus::ValueType<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<Morpheus::ValueType<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<Morpheus::ValueType<long long>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_value_type_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_value_type_v<std::vector<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_value_type_v<std::vector<std::string>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_same_value_type checks if two types have the same value_type
 * i.e they are both scalars.
 *
 */
TEST(TypeTraitsTest, IsSameValueType) {
  bool res = Morpheus::is_same_value_type<Morpheus::ValueType<float>,
                                          Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<float>,
                                     std::vector<float>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<float>, float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<int>,
                                     Morpheus::ValueType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<int>,
                                     std::vector<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<int>, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<double>,
                                     Morpheus::ValueType<double>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<double>,
                                     std::vector<double>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_value_type<Morpheus::ValueType<double>, double>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<long long>,
                                     Morpheus::ValueType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<long long>,
                                     std::vector<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<long long>,
                                     long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<int>,
                                     Morpheus::ValueType<long long>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<double>,
                                     Morpheus::ValueType<long long>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type<Morpheus::ValueType<double>,
                                     Morpheus::ValueType<float>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<float>,
                                       Morpheus::ValueType<float>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<float>,
                                       std::vector<float>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<float>, float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<int>,
                                       Morpheus::ValueType<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<int>,
                                       std::vector<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<int>, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<double>,
                                       Morpheus::ValueType<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<double>,
                                       std::vector<double>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<double>, double>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<long long>,
                                       Morpheus::ValueType<long long>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<long long>,
                                       std::vector<long long>>;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::is_same_value_type_v<Morpheus::ValueType<long long>, long long>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<int>,
                                       Morpheus::ValueType<long long>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<double>,
                                       Morpheus::ValueType<long long>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_value_type_v<Morpheus::ValueType<double>,
                                       Morpheus::ValueType<float>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_index_type checks if the type has a valid index type i.e a
 * \p index_type member trait that is of integral type.
 *
 */
TEST(TypeTraitsTest, IsIndexType) {
  struct TestIndex {
    using index_type = double;
  };
  bool res = Morpheus::is_index_type<TestIndex>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type<Morpheus::IndexType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type<Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type<float>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type<std::vector<double>>::value;
  EXPECT_EQ(res, 0);

  // std::vector doesn't have an index_type but it's value_type is int
  res = Morpheus::is_index_type<std::vector<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type<std::vector<std::string>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<TestIndex>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<Morpheus::IndexType<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type_v<Morpheus::IndexType<long long>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_index_type_v<float>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<std::vector<double>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<std::vector<long long>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_index_type_v<std::vector<std::string>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_same_index_type checks if two types have the same index_type
 * i.e they are both integrals.
 *
 */
TEST(TypeTraitsTest, IsSameIndexType) {
  bool res = Morpheus::is_same_index_type<Morpheus::IndexType<int>,
                                          Morpheus::IndexType<int>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<int>,
                                     std::vector<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<int>, int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<long long>,
                                     Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<long long>,
                                     std::vector<long long>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<long long>,
                                     long long>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type<Morpheus::IndexType<int>,
                                     Morpheus::IndexType<long long>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<int>,
                                       Morpheus::IndexType<int>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<int>,
                                       std::vector<int>>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<int>, int>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<long long>,
                                       Morpheus::IndexType<long long>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<long long>,
                                       std::vector<long long>>;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::is_same_index_type_v<Morpheus::IndexType<long long>, long long>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_index_type_v<Morpheus::IndexType<int>,
                                       Morpheus::IndexType<long long>>;
  EXPECT_EQ(res, 0);
}

TEST(TypeTraitsTest, IsCompatible) {
  struct TestStruct {
    using index_type   = int;
    using value_type   = double;
    using memory_space = Kokkos::HostSpace;
    using array_layout = Kokkos::LayoutRight;
  };

  bool res = Morpheus::is_compatible<TestStruct, TestStruct>::value;
  EXPECT_EQ(res, 1);

  EXPECT_EQ(0, 1);
}

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

  EXPECT_EQ(0, 1);
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

TEST(TypeTraitsTest, IsKokkosSpace) { EXPECT_EQ(0, 1); }
}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
