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

}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
