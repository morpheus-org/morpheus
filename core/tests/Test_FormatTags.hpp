/**
 * Test_FormatTags.hpp
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

#ifndef TEST_CORE_TEST_FORMATTAGS_HPP
#define TEST_CORE_TEST_FORMATTAGS_HPP

#include <Morpheus_Core.hpp>

namespace Test {

/**
 * @brief The \p is_coo_matrix_format_container checks if the passed type is COO
 * Sparse Matrix Format Container. For the check to be valid, the type must be a
 * valid matrix container and has a \p CooFormatTag as a tag member trait.
 *
 */
TEST(FormatTagsTest, IsCooMatrixFormatContainer) {
  struct A {
    using tag = typename Morpheus::MatrixFormatTag<Morpheus::CooFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_coo_matrix_format_container<A>::value));
  EXPECT_TRUE((Morpheus::is_coo_matrix_format_container<
               Morpheus::MatrixFormatTag<Morpheus::CooFormatTag>>::value));

  EXPECT_FALSE(
      (Morpheus::is_coo_matrix_format_container<
          typename Morpheus::MatrixFormatTag<Morpheus::CooFormatTag>::tag>::
           value));
  EXPECT_FALSE((
      Morpheus::is_coo_matrix_format_container<Morpheus::CooFormatTag>::value));

  // Check if is a valid matrix container
  EXPECT_TRUE((Morpheus::is_matrix_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_coo_matrix_format_container_v<
               Morpheus::MatrixFormatTag<Morpheus::CooFormatTag>>));

  EXPECT_FALSE(
      (Morpheus::is_coo_matrix_format_container_v<
          typename Morpheus::MatrixFormatTag<Morpheus::CooFormatTag>::tag>));
}

/**
 * @brief The \p is_csr_matrix_format_container checks if the passed type is CSR
 * Sparse Matrix Format Container. For the check to be valid, the type must be a
 * valid matrix container and has a \p CsrFormatTag as a tag member trait.
 *
 */
TEST(FormatTagsTest, IsCsrMatrixFormatContainer) {
  struct A {
    using tag = typename Morpheus::MatrixFormatTag<Morpheus::CsrFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_csr_matrix_format_container<A>::value));
  EXPECT_TRUE((Morpheus::is_csr_matrix_format_container<
               Morpheus::MatrixFormatTag<Morpheus::CsrFormatTag>>::value));

  EXPECT_FALSE(
      (Morpheus::is_csr_matrix_format_container<
          typename Morpheus::MatrixFormatTag<Morpheus::CsrFormatTag>::tag>::
           value));
  EXPECT_FALSE((
      Morpheus::is_csr_matrix_format_container<Morpheus::CsrFormatTag>::value));

  // Check if is a valid matrix container
  EXPECT_TRUE((Morpheus::is_matrix_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_csr_matrix_format_container_v<
               Morpheus::MatrixFormatTag<Morpheus::CsrFormatTag>>));
  EXPECT_FALSE(
      (Morpheus::is_csr_matrix_format_container_v<
          typename Morpheus::MatrixFormatTag<Morpheus::CsrFormatTag>::tag>));
}

/**
 * @brief The \p is_dia_matrix_format_container checks if the passed type is DIA
 * Sparse Matrix Format Container. For the check to be valid, the type must be a
 * valid matrix container and has a \p DiaFormatTag as a tag member trait.
 *
 */
TEST(FormatTagsTest, IsDiaMatrixFormatContainer) {
  struct A {
    using tag = typename Morpheus::MatrixFormatTag<Morpheus::DiaFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_dia_matrix_format_container<A>::value));
  EXPECT_TRUE((Morpheus::is_dia_matrix_format_container<
               Morpheus::MatrixFormatTag<Morpheus::DiaFormatTag>>::value));

  EXPECT_FALSE(
      (Morpheus::is_dia_matrix_format_container<
          typename Morpheus::MatrixFormatTag<Morpheus::DiaFormatTag>::tag>::
           value));

  EXPECT_FALSE((
      Morpheus::is_dia_matrix_format_container<Morpheus::DiaFormatTag>::value));

  // Check if is a valid matrix container
  EXPECT_TRUE((Morpheus::is_matrix_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_dia_matrix_format_container_v<
               Morpheus::MatrixFormatTag<Morpheus::DiaFormatTag>>));

  EXPECT_FALSE(
      (Morpheus::is_dia_matrix_format_container_v<
          typename Morpheus::MatrixFormatTag<Morpheus::DiaFormatTag>::tag>));
}

/**
 * @brief The \p is_dynamic_matrix_format_container checks if the passed type is
 * Dynamic Sparse Matrix Format Container. For the check to be valid, the type
 * must be a valid matrix container and has a \p DynamicMatrixFormatTag as a tag
 * member trait.
 *
 */
TEST(FormatTagsTest, IsDynamicMatrixFormatContainer) {
  struct A {
    using tag = typename Morpheus::MatrixFormatTag<
        Morpheus::DynamicMatrixFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_dynamic_matrix_format_container<A>::value));

  EXPECT_TRUE(
      (Morpheus::is_dynamic_matrix_format_container<
          Morpheus::MatrixFormatTag<Morpheus::DynamicMatrixFormatTag>>::value));

  EXPECT_FALSE((Morpheus::is_dynamic_matrix_format_container<
                typename Morpheus::MatrixFormatTag<
                    Morpheus::DynamicMatrixFormatTag>::tag>::value));

  EXPECT_FALSE((Morpheus::is_dynamic_matrix_format_container<
                Morpheus::DynamicMatrixFormatTag>::value));

  // Check if is a valid matrix container
  EXPECT_TRUE((Morpheus::is_matrix_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_dynamic_matrix_format_container_v<
               Morpheus::MatrixFormatTag<Morpheus::DynamicMatrixFormatTag>>));

  EXPECT_FALSE((Morpheus::is_dynamic_matrix_format_container_v<
                typename Morpheus::MatrixFormatTag<
                    Morpheus::DynamicMatrixFormatTag>::tag>));
}

/**
 * @brief The \p is_dense_matrix_format_container checks if the passed type is
 * Dense Matrix Format Container. For the check to be valid, the type must be a
 * valid matrix container and has a \p DenseMatrixFormatTag as a tag member
 * trait.
 *
 */
TEST(FormatTagsTest, IsDenseMatrixFormatContainer) {
  struct A {
    using tag =
        typename Morpheus::MatrixFormatTag<Morpheus::DenseMatrixFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_dense_matrix_format_container<A>::value));

  EXPECT_TRUE(
      (Morpheus::is_dense_matrix_format_container<
          Morpheus::MatrixFormatTag<Morpheus::DenseMatrixFormatTag>>::value));

  EXPECT_FALSE((Morpheus::is_dense_matrix_format_container<
                typename Morpheus::MatrixFormatTag<
                    Morpheus::DenseMatrixFormatTag>::tag>::value));

  EXPECT_FALSE((Morpheus::is_dense_matrix_format_container<
                Morpheus::DenseMatrixFormatTag>::value));

  // Check if is a valid matrix container
  EXPECT_TRUE((Morpheus::is_matrix_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_dense_matrix_format_container_v<
               Morpheus::MatrixFormatTag<Morpheus::DenseMatrixFormatTag>>));

  EXPECT_FALSE((Morpheus::is_dense_matrix_format_container_v<
                typename Morpheus::MatrixFormatTag<
                    Morpheus::DenseMatrixFormatTag>::tag>));
}

/**
 * @brief The \p is_dense_vector_format_container checks if the passed type is
 * Dense Vector Format Container. For the check to be valid, the type must be a
 * valid matrix container and has a \p DenseVectorFormatTag as a tag member
 * trait.
 *
 */
TEST(FormatTagsTest, IsDenseVectorFormatContainer) {
  struct A {
    using tag =
        typename Morpheus::VectorFormatTag<Morpheus::DenseVectorFormatTag>::tag;
  };

  EXPECT_TRUE((Morpheus::is_dense_vector_format_container<A>::value));

  EXPECT_TRUE(
      (Morpheus::is_dense_vector_format_container<
          Morpheus::VectorFormatTag<Morpheus::DenseVectorFormatTag>>::value));

  EXPECT_FALSE((Morpheus::is_dense_vector_format_container<
                typename Morpheus::VectorFormatTag<
                    Morpheus::DenseVectorFormatTag>::tag>::value));

  EXPECT_FALSE((Morpheus::is_dense_vector_format_container<
                Morpheus::DenseVectorFormatTag>::value));

  // Check if is a valid vector container
  EXPECT_TRUE((Morpheus::is_vector_container<A>::value));

  // Type alias
  EXPECT_TRUE((Morpheus::is_dense_vector_format_container_v<
               Morpheus::VectorFormatTag<Morpheus::DenseVectorFormatTag>>));

  EXPECT_FALSE((Morpheus::is_dense_vector_format_container_v<
                typename Morpheus::VectorFormatTag<
                    Morpheus::DenseVectorFormatTag>::tag>));
}

}  // namespace Test

#endif  // TEST_CORE_TEST_TYPETRAITS_HPP
