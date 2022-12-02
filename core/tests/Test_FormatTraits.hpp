/**
 * Test_FormatTraits.hpp
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

#ifndef TEST_CORE_TEST_FORMAT_TRAITS_HPP
#define TEST_CORE_TEST_FORMAT_TRAITS_HPP

#include <Morpheus_Core.hpp>

namespace Impl {
template <typename T>
struct with_tag {
  using tag = T;
};

struct no_traits {};

// used in IsCompatible & IsDynamicallyCompatible
template <typename T1, typename T2, typename T3, typename T4,
          typename Tag = void>
struct TestStruct {
  using value_type   = T1;
  using index_type   = T2;
  using memory_space = typename T3::memory_space;
  using array_layout = T4;
  using tag          = Tag;
};

}  // namespace Impl

namespace Test {
/**
 * @brief The \p is_matrix_container checks if the passed type is a valid
 Matrix
 * container. For the check to be valid the type must have a valid Matrix tag
 * trait. Note that both dense, sparse and dynamic matrices should be valid
 * matrix containers.
 *
 */
TEST(FormatTraitsTest, IsMatrixContainer) {
  bool res = Morpheus::is_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is a valid tag for Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Sparse Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dynamic Matrix Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A COO Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Matrix Container
  res = Morpheus::is_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::is_matrix_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Matrix Container
  res = Morpheus::is_matrix_container_v<Impl::with_tag<Morpheus::CooFormatTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_sparse_matrix_container checks if the passed type is a
 valid
 * Sparse Matrix container. For the check to be valid the type must have a
 valid
 * Sparse Matrix tag trait. Note that all supported sparse matrix storage
 * formats should be valid Sparse Matrix Containers.
 *
 */
TEST(FormatTraitsTest, IsSparseMatrixContainer) {
  bool res = Morpheus::is_sparse_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_sparse_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Sparse Matrix Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Matrix Tag is not a valid tag for Sparse Matrix Container - A sparse
  // matrix is a matrix but a matrix is not necessarilly sparse
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Format Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Vector Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  // A Sparse Matrix Tag is a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is not a valid tag for Sparse Matrix Container
  res = Morpheus::is_sparse_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_dense_matrix_container checks if the passed type is a
 valid
 * Dense Matrix container. For the check to be valid the type must have a
 valid
 * Dense Matrix tag trait. Note that all supported dense matrix
 representations
 * should be valid Dense Matrix Containers.
 *
 */
TEST(FormatTraitsTest, IsDenseMatrixContainer) {
  bool res = Morpheus::is_dense_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dense_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Dense Matrix Container - A dense
  // matrix is a matrix but a matrix is not necessarilly dense
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  // A Dense Matrix Tag is a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>;
  EXPECT_EQ(res, 1);

  // A Vector Tag is not a valid tag for Dense Matrix Container
  res = Morpheus::is_dense_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::VectorTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_vector_container checks if the passed type is a valid
 Vector
 * container. For the check to be valid the type must have a valid Vector tag
 * trait. Note that both dense and sparse vectors should be valid
 * vector containers.
 *
 */
TEST(FormatTraitsTest, IsVectorContainer) {
  bool res = Morpheus::is_vector_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_vector_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::VectorTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  // A Sparse Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::SparseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A COO Tag is a not valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Vector Container
  res = Morpheus::is_vector_container<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  // A Dense Vector Tag is a valid tag for Vector Container
  res = Morpheus::is_vector_container_v<
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>;
  EXPECT_EQ(res, 1);

  // A COO Tag is a not valid tag for Vector Container
  res = Morpheus::is_vector_container_v<Impl::with_tag<Morpheus::CooFormatTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_container checks if the passed type is a valid Morpheus
 Container. For the check to be valid, the type should either be a valid
 Matrix or Vector container.
 *
 */
TEST(FormatTraitsTest, IsContainer) {
  bool res = Morpheus::is_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<
      Impl::with_tag<Morpheus::Impl::DenseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<
      Impl::with_tag<Morpheus::Impl::DenseVectorTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Matrix Format Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // A Dense Vector Format Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container<
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::is_container_v<Impl::with_tag<int>>;
  EXPECT_EQ(res, 0);

  // A COO Tag is a valid tag for Morpheus Container
  res = Morpheus::is_container_v<Impl::with_tag<Morpheus::CooFormatTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_dynamic_matrix_container checks if the passed type is a
 * valid Dynamic Matrix Container. For the check to be valid, the type must
 be a
 * valid Matrix container and have a valid Dynamic Matrix tag.
 *
 */
TEST(FormatTraitsTest, IsDynamicContainer) {
  bool res = Morpheus::is_dynamic_matrix_container<Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamic_matrix_container<Impl::no_traits>::value;
  EXPECT_EQ(res, 0);

  // A COO Tag is a not valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Matrix Tag is  a valid Matrix Container but a Matrix Container is not
  // necessarily Dynamic
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Sparse Matrix Tag is  a valid Matrix Container but a Sparse Matrix
  // Container is not necessarily Dynamic
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Matrix Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dense Vector Tag is not a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // A Dynamic Tag is a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container<
      Impl::with_tag<Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  // A Sparse Matrix Tag is a valid Matrix Container but a Sparse Matrix
  // Container is not necessarily Dynamic
  res = Morpheus::is_dynamic_matrix_container_v<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>;
  EXPECT_EQ(res, 0);

  // A Dynamic Tag is a valid tag for Dynamic Matrix Container
  res = Morpheus::is_dynamic_matrix_container_v<
      Impl::with_tag<Morpheus::DynamicMatrixFormatTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p has_same_format checks if the two types passed have the same
 * storage format. For the check to be valid, both types must be valid
 Morpheus
 * Containers and have the same tag.
 *
 */
TEST(FormatTraitsTest, HasSameFormat) {
  // Same types
  bool res =
      Morpheus::has_same_format<Impl::with_tag<Morpheus::CooFormatTag>,
                                Impl::with_tag<Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::DynamicMatrixFormatTag>,
      Impl::with_tag<Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::DenseVectorFormatTag>,
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>,
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 1);

  // Check against derived formats
  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::CooFormatTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::CooFormatTag>,
      Impl::with_tag<Morpheus::Impl::MatrixTag>>::value;
  EXPECT_EQ(res, 0);

  // Check against different formats
  res =
      Morpheus::has_same_format<Impl::with_tag<Morpheus::CooFormatTag>,
                                Impl::with_tag<Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::has_same_format<Impl::with_tag<Morpheus::DenseMatrixFormatTag>,
                                Impl::with_tag<Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_format<
      Impl::with_tag<Morpheus::DenseMatrixFormatTag>,
      Impl::with_tag<Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Morpheus format against built-in type
  res =
      Morpheus::has_same_format<Impl::with_tag<Morpheus::DenseMatrixFormatTag>,
                                Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  // Built-in types
  res = Morpheus::has_same_format<Impl::with_tag<int>,
                                  Impl::with_tag<int>>::value;
  EXPECT_EQ(res, 0);

  /* Testing Alias */
  res = Morpheus::has_same_format_v<
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>;
  EXPECT_EQ(res, 1);

  // Check against derived formats
  res = Morpheus::has_same_format_v<
      Impl::with_tag<Morpheus::CooFormatTag>,
      Impl::with_tag<Morpheus::Impl::SparseMatrixTag>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_compatible checks if the types passed are compatible
 * containers. For the check to be valid, the types must have the same memory
 * space, layout, value and index types.
 *
 */
TEST(FormatTraitsTest, IsCompatible) {
  bool res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_compatible<
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_compatible<
      Impl::TestStruct<double, long long, Morpheus::HostSpace,
                       Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutLeft>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_compatible_v<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_compatible_v<
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_dynamically_compatible checks if the types passed are
 * dynamically compatible containers. For the check to be valid, the types
 * must be compatible containers and at least one of the two also be a Dynamic
 * Matrix Container.
 *
 */
TEST(FormatTraitsTest, IsDynamicallyCompatible) {
  // Compatible types but none is dynamic container
  bool res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  // Compatible and second container is dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Compatible and both dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Both dynamic but not compatible
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

#endif

  /* Testing Alias */
  res = Morpheus::is_dynamically_compatible_v<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_dynamically_compatible_v<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_format_compatible checks if the types passed are format
 * compatible containers. For the check to be valid, the types must be
 * compatible containers and also have the same format.
 *
 */
TEST(FormatTraitsTest, IsFormatCompatible) {
  // Compatible types but not same format
  bool res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Compatible types but invalid format
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Morpheus::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and Same Format
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseVectorFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Both same format but not compatible
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, long long, Morpheus::HostSpace,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  // Compatible and first container is dynamic
  res = Morpheus::is_format_compatible_v<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>;
  EXPECT_EQ(res, 0);

  // Compatible and Same Format
  res = Morpheus::is_format_compatible_v<
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_format_compatible_different_space checks if the types
 * passed are format compatible containers but from different memory space.
 * For the check to be valid, the types must have the same format, layout,
 * value and index types but different memory space.
 *
 */
TEST(FormatTraitsTest, IsFormatCompatibleDifferentSpace) {
  // Same format, layout, value and index types - Same Space
  bool res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  // Same format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Different format, same layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different layout, same format, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different value type, same format layout and index type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different index type, same format layout and value type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long, Morpheus::HostSpace,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  // Same format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Different format, same layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different layout, same format, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different value type, same format layout and index type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different index type, same format layout and value type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long, Morpheus::HostSpace,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

  // Different format, same layout, value and index types - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different layout, same format, value and index types - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutLeft, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutLeft, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutLeft, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different value type, same format layout and index type - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different index type, same format layout and value type - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long,
                       typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long,
                       typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Dynamic format, layout, value and index types - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, typename TEST_CUSTOM_SPACE::memory_space,
                       Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, typename TEST_GENERIC_SPACE::memory_space,
                       Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  // Dynamic format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_HIP)
  // Dynamic format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Morpheus::HIPSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Morpheus::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);
#endif
}

}  // namespace Test

#endif  // TEST_CORE_TEST_FORMAT_TRAITS_HPP
