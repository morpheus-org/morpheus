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

namespace Impl {
template <typename T>
struct with_tag {
  using tag = T;
};

template <typename T>
struct with_memspace {
  using memory_space = T;
};

template <typename T>
struct with_layout {
  using array_layout = T;
};

struct no_traits {};

// used in IsCompatible & IsDynamicallyCompatible
template <typename T1, typename T2, typename T3, typename T4,
          typename Tag = void>
struct TestStruct {
  using value_type   = T1;
  using index_type   = T2;
  using memory_space = T3;
  using array_layout = T4;
  using tag          = Tag;
};

}  // namespace Impl

namespace Test {

/**
 * @brief The \p is_variant_member_v checks if the passed type is a member of
 * the variant container
 *
 */
TEST(TypeTraitsTest, IsVariantMember) {
  using variant = Morpheus::Impl::Variant::variant<int, double, float>;

  struct A {};
  bool res = Morpheus::is_variant_member_v<double, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<float, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<int, variant>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_variant_member_v<long long, variant>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_variant_member_v<char, variant>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_variant_member_v<A, variant>;
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
 * @brief The \p is_matrix_container checks if the passed type is a valid Matrix
 * container. For the check to be valid the type must have a valid Matrix tag
 * trait. Note that both dense, sparse and dynamic matrices should be valid
 * matrix containers.
 *
 */
TEST(TypeTraitsTest, IsMatrixContainer) {
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
 * @brief The \p is_sparse_matrix_container checks if the passed type is a valid
 * Sparse Matrix container. For the check to be valid the type must have a valid
 * Sparse Matrix tag trait. Note that all supported sparse matrix storage
 * formats should be valid Sparse Matrix Containers.
 *
 */
TEST(TypeTraitsTest, IsSparseMatrixContainer) {
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
 * @brief The \p is_dense_matrix_container checks if the passed type is a valid
 * Dense Matrix container. For the check to be valid the type must have a valid
 * Dense Matrix tag trait. Note that all supported dense matrix representations
 * should be valid Dense Matrix Containers.
 *
 */
TEST(TypeTraitsTest, IsDenseMatrixContainer) {
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
 * @brief The \p is_vector_container checks if the passed type is a valid Vector
 * container. For the check to be valid the type must have a valid Vector tag
 * trait. Note that both dense and sparse vectors should be valid
 * vector containers.
 *
 */
TEST(TypeTraitsTest, IsVectorContainer) {
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
 Container. For the check to be valid, the type should either be a valid Matrix
 or Vector container.
 *
 */
TEST(TypeTraitsTest, IsContainer) {
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
 * valid Dynamic Matrix Container. For the check to be valid, the type must be a
 * valid Matrix container and have a valid Dynamic Matrix tag.
 *
 */
TEST(TypeTraitsTest, IsDynamicContainer) {
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
 * storage format. For the check to be valid, both types must be valid Morpheus
 * Containers and have the same tag.
 *
 */
TEST(TypeTraitsTest, HasSameFormat) {
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
 * @brief The \p is_memory_space checks if the passed type is a valid memory
 * space. For the check to be valid, the type must be one of the supported
 * memory spaces.
 *
 */
TEST(TypeTraitsTest, IsMemorySpace) {
  bool res = Morpheus::is_memory_space<Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_memory_space<int>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_memory_space<Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);
#endif

  // Kokkos Execution Space
  res = Morpheus::is_memory_space<TEST_EXECSPACE>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_memory_space<typename TEST_EXECSPACE::memory_space>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::is_memory_space_v<Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_memory_space_v<int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_memory_space checks if the passed type has a valid memory
 * space. For the check to be valid, the type must be a valid memory space and
 * have a \p memory_space trait.
 *
 */
TEST(TypeTraitsTest, HasMemorySpace) {
  // A structure like this meets the requirements of a valid memory space i.e
  // has a memory_space trait that is the same as it's name BUT this is not
  // supported as a MemorySpace
  bool res = Morpheus::has_memory_space<Impl::with_memspace<int>>::value;
  EXPECT_EQ(res, 0);

  // HostSpace has a memory_space trait that we support
  res = Morpheus::has_memory_space<Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

  // HostSpace is also itself a memory space
  res =
      Morpheus::has_memory_space<Impl::with_memspace<Kokkos::HostSpace>>::value;
  EXPECT_EQ(res, 1);

#if defined(MORPHEUS_ENABLE_CUDA)
  // CudaSpace has a memory_space trait that we support
  res = Morpheus::has_memory_space<Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);

  // CudaSpace is also itself a memory space
  res =
      Morpheus::has_memory_space<Impl::with_memspace<Kokkos::CudaSpace>>::value;
  EXPECT_EQ(res, 1);
#endif

  res = Morpheus::has_memory_space<TEST_EXECSPACE>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::has_memory_space<typename TEST_EXECSPACE::memory_space>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::has_memory_space_v<Impl::with_memspace<int>>;
  EXPECT_EQ(res, 0);

  // HostSpace has a memory_space trait that we support
  res = Morpheus::has_memory_space_v<Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_same_memory_space checks if the two types passed are the
 * same memory space. For the check to be valid, both types must be a valid
 * memory space and be the same.
 *
 */
TEST(TypeTraitsTest, IsSameMemorySpace) {
  bool res = Morpheus::is_same_memory_space<Kokkos::HostSpace,
                                            Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_same_memory_space<int, int>::value;
  EXPECT_EQ(res, 0);

  // Built-in type with valid memory space
  res = Morpheus::is_same_memory_space<int, Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_same_memory_space<Kokkos::CudaSpace,
                                       Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);
#endif

  // Execution Space
  res = Morpheus::is_same_memory_space<TEST_EXECSPACE, TEST_EXECSPACE>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_same_memory_space<
      typename TEST_EXECSPACE::memory_space,
      typename TEST_EXECSPACE::memory_space>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::is_same_memory_space_v<Kokkos::HostSpace, Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::is_same_memory_space_v<int, int>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p has_same_memory_space checks if the two types passed have the
 * same memory space. For the check to be valid, both types must have a
 * \p memory_space trait and the \p is_same_memory_space must be satisfied.
 *
 */
TEST(TypeTraitsTest, HasSameMemorySpace) {
  bool res = Morpheus::has_same_memory_space<Kokkos::HostSpace,
                                             Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::has_same_memory_space<int, int>::value;
  EXPECT_EQ(res, 0);

  // Built-in type with valid memory space
  res = Morpheus::has_same_memory_space<int, Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_same_memory_space<Impl::with_memspace<int>,
                                        Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::has_same_memory_space<Kokkos::CudaSpace,
                                        Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 1);
#endif

  // Execution Space
  res = Morpheus::has_same_memory_space<TEST_EXECSPACE, TEST_EXECSPACE>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_same_memory_space<
      typename TEST_EXECSPACE::memory_space,
      typename TEST_EXECSPACE::memory_space>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::has_same_memory_space_v<Kokkos::HostSpace, Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);

  // Built-in type
  res = Morpheus::has_same_memory_space_v<int, int>;
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
 * @brief The \p is_compatible checks if the types passed are compatible
 * containers. For the check to be valid, the types must have the same memory
 * space, layout, value and index types.
 *
 */
TEST(TypeTraitsTest, IsCompatible) {
  bool res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_compatible<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_compatible<
      Impl::TestStruct<double, long long, Kokkos::HostSpace,
                       Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutLeft>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_compatible<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_compatible_v<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_compatible_v<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);
}

/**
 * @brief The \p is_dynamically_compatible checks if the types passed are
 * dynamically compatible containers. For the check to be valid, the types
 * must be compatible containers and at least one of the two also be a Dynamic
 * Matrix Container.
 *
 */
TEST(TypeTraitsTest, IsDynamicallyCompatible) {
  // Compatible types but none is dynamic container
  bool res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 1);

  // Compatible and second container is dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Compatible and both dynamic
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Both dynamic but not compatible
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, long long, Kokkos::HostSpace,
                       Kokkos::LayoutRight, Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_dynamically_compatible<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_dynamically_compatible_v<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_dynamically_compatible_v<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_format_compatible checks if the types passed are format
 * compatible containers. For the check to be valid, the types must be
 * compatible containers and also have the same format.
 *
 */
TEST(TypeTraitsTest, IsFormatCompatible) {
  // Compatible types but not same format
  bool res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Compatible types but invalid format
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight>,
      Impl::TestStruct<double, int, Kokkos::HostSpace,
                       Kokkos::LayoutRight>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and first container is dynamic
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Compatible and Same Format
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseVectorFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseVectorFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DenseMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Both same format but not compatible
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, long long, Kokkos::HostSpace,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_format_compatible<
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_format_compatible<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  // Compatible and first container is dynamic
  res = Morpheus::is_format_compatible_v<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>;
  EXPECT_EQ(res, 0);

  // Compatible and Same Format
  res = Morpheus::is_format_compatible_v<
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
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
TEST(TypeTraitsTest, IsFormatCompatibleDifferentSpace) {
  // Same format, layout, value and index types - Same Space
  bool res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_CUDA)
  // Same format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 1);

  // Different format, same layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CsrFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different layout, same format, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutLeft,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different value type, same format layout and index type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different index type, same format layout and value type - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long, Kokkos::HostSpace,
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

  // Different layout, same format, value and index types - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutLeft, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different value type, same format layout and index type - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<float, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>>::value;
  EXPECT_EQ(res, 0);

  // Different index type, same format layout and value type - Same Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, typename TEST_EXECSPACE::memory_space,
                       Kokkos::LayoutRight, Morpheus::CooFormatTag>,
      Impl::TestStruct<double, long long, typename TEST_EXECSPACE::memory_space,
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

#if defined(MORPHEUS_ENABLE_CUDA)
  // Dynamic format, layout, value and index types - Different Space
  res = Morpheus::is_format_compatible_different_space<
      Impl::TestStruct<double, int, Kokkos::CudaSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>,
      Impl::TestStruct<double, int, Kokkos::HostSpace, Kokkos::LayoutRight,
                       Morpheus::DynamicMatrixFormatTag>>::value;
  EXPECT_EQ(res, 1);
#endif
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

/**
 * @brief The \p is_execution_space checks if the passed type is a valid
 * executions space. For the check to be valid, the type must be one of the
 * supported execution spaces.
 *
 */
TEST(TypeTraitsTest, IsExecutionSpace) {
  struct A {};
  bool res = Morpheus::is_execution_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_execution_space<A>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_execution_space<Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_execution_space<Kokkos::DefaultExecutionSpace>::value;
  EXPECT_EQ(res, 1);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_execution_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_execution_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_execution_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 1);
#endif

  /* Testing Alias */
  res = Morpheus::is_execution_space_v<A>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_host_memory_space checks if the passed type is a valid
 * Host memory space. For the check to be valid, the type must be one of the
 * supported Host memory spaces.
 *
 */
TEST(TypeTraitsTest, IsHostMemorySpace) {
  // A structure like this meets the requirements of a valid memory space i.e
  // has a memory_space trait that is the same as it's name BUT this is not
  // supported as a MemorySpace
  struct TestSpace {
    using memory_space = TestSpace;
  };

  bool res = Morpheus::is_host_memory_space<TestSpace>::value;
  EXPECT_EQ(res, 0);

  // Built-in type
  res = Morpheus::is_host_memory_space<int>::value;
  EXPECT_EQ(res, 0);

  // Valid Memory Space
  res = Morpheus::is_host_memory_space<Kokkos::HostSpace>::value;
  EXPECT_EQ(res, 1);

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_host_memory_space<Kokkos::CudaSpace>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  // Built-in type
  res = Morpheus::is_host_memory_space_v<int>;
  EXPECT_EQ(res, 0);

  // Valid Memory Space
  res = Morpheus::is_host_memory_space_v<Kokkos::HostSpace>;
  EXPECT_EQ(res, 1);
}

/**
 * @brief The \p is_host_execution_space checks if the passed type is a valid
 * Host executions space. For the check to be valid, the type must be one of
 * the supported Host execution spaces.
 *
 */
TEST(TypeTraitsTest, IsHostExecutionSpace) {
  struct A {};
  bool res = Morpheus::is_host_execution_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_host_execution_space<A>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_host_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 1);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_host_execution_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_host_execution_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_host_execution_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_host_execution_space_v<A>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_host_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 1);
}

#if defined(MORPHEUS_ENABLE_SERIAL)
/**
 * @brief The \p is_serial_execution_space checks if the passed type is a
 * valid Serial executions space. For the check to be valid, the type must be
 * a Serial execution space.
 *
 */
TEST(TypeTraitsTest, IsSerialExecutionSpace) {
  struct A {};
  bool res = Morpheus::is_serial_execution_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_serial_execution_space<A>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_serial_execution_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::is_serial_execution_space<
      Morpheus::GenericSpace<Kokkos::Serial>>::value;
  EXPECT_EQ(res, 1);

#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_serial_execution_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_serial_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 0);
#elif defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_serial_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_serial_execution_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_serial_execution_space_v<A>;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_OPENMP)
  res =
      Morpheus::is_serial_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 0);
#elif defined(MORPHEUS_ENABLE_SERIAL)
  res =
      Morpheus::is_serial_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 1);
#endif
}
#endif  // MORPHEUS_ENABLE_SERIAL

#if defined(MORPHEUS_ENABLE_OPENMP)
/**
 * @brief The \p is_openmp_execution_space checks if the passed type is a
 * valid OpenMP executions space. For the check to be valid, the type must be
 * a OpenMP execution space.
 *
 */
TEST(TypeTraitsTest, IsOpenMPExecutionSpace) {
  struct A {};
  bool res = Morpheus::is_openmp_execution_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_openmp_execution_space<A>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_openmp_execution_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_openmp_execution_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 1);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_openmp_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 1);
#elif defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_openmp_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::is_openmp_execution_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 0);
#endif

  /* Testing Alias */
  res = Morpheus::is_openmp_execution_space_v<A>;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_OPENMP)
  res =
      Morpheus::is_openmp_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 1);
#elif defined(MORPHEUS_ENABLE_SERIAL)
  res =
      Morpheus::is_openmp_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 0);
#endif
}
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
/**
 * @brief The \p is_cuda_execution_space checks if the passed type is a valid
 * Cuda executions space. For the check to be valid, the type must be a Cuda
 * execution space.
 *
 */
TEST(TypeTraitsTest, IsCudaExecutionSpace) {
  struct A {};
  bool res = Morpheus::is_cuda_execution_space<int>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_cuda_execution_space<A>::value;
  EXPECT_EQ(res, 0);

#if defined(MORPHEUS_ENABLE_SERIAL)
  res = Morpheus::is_cuda_execution_space<Kokkos::Serial>::value;
  EXPECT_EQ(res, 0);
#endif

#if defined(MORPHEUS_ENABLE_OPENMP)
  res = Morpheus::is_cuda_execution_space<Kokkos::OpenMP>::value;
  EXPECT_EQ(res, 0);
#endif

  res = Morpheus::is_cuda_execution_space<
      Kokkos::DefaultHostExecutionSpace>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_cuda_execution_space<Kokkos::Cuda>::value;
  EXPECT_EQ(res, 1);

  /* Testing Alias */
  res = Morpheus::is_cuda_execution_space_v<A>;
  EXPECT_EQ(res, 0);

  res = Morpheus::is_cuda_execution_space_v<Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(res, 0);
}
#endif  // MORPHEUS_ENABLE_CUDA

/**
 * @brief The \p has_access checks if first type (ExecutionSpace) has access
 * to the arbitrary number of types passed. For the check to be valid, the
 * first type must be a valid execution space, the rest must have a valid
 * memory space and the execution space must be able to access them.
 *
 */
TEST(TypeTraitsTest, HasAccess) {
  bool res =
      Morpheus::has_access<Kokkos::Serial,
                           Impl::with_memspace<Kokkos::HostSpace>>::value;
  EXPECT_EQ(res, 1);

  res = Morpheus::has_access<Kokkos::Serial,
                             Impl::with_memspace<Kokkos::HostSpace>,
                             Impl::with_memspace<Kokkos::HostSpace>,
                             Impl::with_memspace<Kokkos::HostSpace>,
                             Impl::with_memspace<Kokkos::HostSpace>>::value;

#if defined(MORPHEUS_ENABLE_CUDA)
  res = Morpheus::has_access<Kokkos::Cuda,
                             Impl::with_memspace<Kokkos::HostSpace>>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::has_access<Kokkos::Cuda, Impl::with_memspace<Kokkos::HostSpace>,
                           Impl::with_memspace<Kokkos::HostSpace>,
                           Impl::with_memspace<Kokkos::HostSpace>,
                           Impl::with_memspace<Kokkos::HostSpace>>::value;
  EXPECT_EQ(res, 0);

  res = Morpheus::has_access<Kokkos::Cuda,
                             Impl::with_memspace<Kokkos::CudaSpace>>::value;
  EXPECT_EQ(res, 1);

  res =
      Morpheus::has_access<Kokkos::Cuda, Impl::with_memspace<Kokkos::HostSpace>,
                           Impl::with_memspace<Kokkos::CudaSpace>,
                           Impl::with_memspace<Kokkos::HostSpace>,
                           Impl::with_memspace<Kokkos::HostSpace>>::value;
  EXPECT_EQ(res, 0);

  res =
      Morpheus::has_access<Kokkos::Cuda, Impl::with_memspace<Kokkos::CudaSpace>,
                           Impl::with_memspace<Kokkos::CudaSpace>,
                           Impl::with_memspace<Kokkos::CudaSpace>,
                           Impl::with_memspace<Kokkos::CudaSpace>>::value;
  EXPECT_EQ(res, 1);
#endif
  EXPECT_EQ(res, 1);
}

}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
