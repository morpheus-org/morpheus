/**
 * Test_DenseMatrix.hpp
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

#ifndef TEST_CORE_TEST_DENSEMATRIX_HPP
#define TEST_CORE_TEST_DENSEMATRIX_HPP

#include <Morpheus_Core.hpp>

#include <setup/DenseMatrixDefinition_Utils.hpp>

// Used for testing unary operations for same type container
template <typename UnaryContainer>
class DenseMatrixUnaryTest : public ::testing::Test {
 public:
  using DenseMatrix = UnaryContainer;
  using HostMirror  = typename UnaryContainer::HostMirror;
  // No need for setup and tear-down in this case, mainly care about the types
  // any setup and tear-down will be made by each individual test
};

// Used for testing behaviour between many types of the same container
template <typename DenseMatrixTypes>
class DenseMatrixTypesTest : public ::testing::Test {
 public:
  using DenseMatrix_v    = typename DenseMatrixTypes::v;
  using DenseMatrix_vl   = typename DenseMatrixTypes::vl;
  using DenseMatrix_vis  = typename DenseMatrixTypes::vis;
  using DenseMatrix_vil  = typename DenseMatrixTypes::vil;
  using DenseMatrix_vils = typename DenseMatrixTypes::vils;
  using DenseMatrix_vls  = typename DenseMatrixTypes::vls;
};

namespace Test {

// TYPED_TEST_CASE(DenseMatrixUnaryTest, DenseMatrixUnary);

// TYPED_TEST(DenseMatrixUnaryTest, Traits) {
//   // Check DenseMatrix Specific Traits:
//   // Tag, value_array_type, value_array_pointer, value_array_reference
//   // Repeat that for the HostMirror too
//   // Check value_array_type traits too
//   // Ensure size is of type size_t and not index_type
//   // Add size_type trait
//   static_assert(std::is_same<typename TestFixture::DenseMatrix::tag,
//                              Morpheus::DenseMatrixTag>::value);
// }

// TYPED_TEST(DenseMatrixUnaryTest, DefaultConstruction) {
//   // DenseMatrix()
// }

// TYPED_TEST(DenseMatrixUnaryTest, NormalConstruction) {
//   // DenseMatrix(const std::string name, index_type n, value_type val = 0)
//   // DenseMatrix(index_type n, value_type val = 0)
//   // TODO: Change n to size_t
// }

// TYPED_TEST(DenseMatrixUnaryTest, RandomConstruction) {
//   // DenseMatrix(const std::string name, index_type n, Generator rand_pool,
//   //             const value_type range_low, const value_type range_high)
//   // TODO: Change n to size_t
// }

// // Changed that to DenseMatrixBinaryTests
// TYPED_TEST(DenseMatrixUnaryTest, ShallowCopySemantics) {
//   // DenseMatrix(const DenseMatrix<VR, PR...>& src, ...)
//   // operator=(const DenseMatrix<VR, PR...>& src)
//   // DenseMatrix(const DenseMatrix&) = default;
//   // DenseMatrix& operator=(const DenseMatrix&) = default;
// }

// // Changed that to DenseMatrixBinaryTests
// TYPED_TEST(DenseMatrixUnaryTest, Allocate) {
//   // DenseMatrix& allocate(const std::string name,
//   //                       const DenseMatrix<VR, PR...>& src)
// }

// TYPED_TEST(DenseMatrixUnaryTest, Assign) {
//   // assign(const index_type n, const value_type val)
//   // TODO: Change n to size_t
// }

// TYPED_TEST(DenseMatrixUnaryTest, AssignRandom) {
//   // assign(Generator rand_pool, const value_type range_low,
//   //        const value_type range_high)
// }

// TYPED_TEST(DenseMatrixUnaryTest, Resize) {
//   // resize(index_type n)
//   // resize(const index_type n, const index_type val)
//   // TODO: Change n to size_t
// }

// TYPED_TEST(DenseMatrixUnaryTest, UtilRoutines) {
//   // size()
//   // data()
//   // view()
//   // const_view()
//   // name()
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_DENSEMATRIX_HPP
