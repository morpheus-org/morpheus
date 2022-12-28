/**
 * Morpheus_MatrixOperations_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_ELL_CUDA_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_ELL_CUDA_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Ell/Kernels/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void update_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using size_type  = typename Matrix::size_type;
  using index_type = typename Matrix::index_type;
  using value_type = typename Matrix::value_type;

  const size_type BLOCK_SIZE = 256;
  const size_type MAX_BLOCKS = max_active_blocks(
      Kernels::update_ell_diagonal_kernel<value_type, index_type, size_type>,
      BLOCK_SIZE, 0);
  const size_type NUM_BLOCKS = std::min<size_type>(
      MAX_BLOCKS, Impl::ceil_div<size_type>(A.nrows(), BLOCK_SIZE));

  const index_type num_entries_per_row = A.ccolumn_indices().ncols();
  const index_type pitch               = A.ccolumn_indices().nrows();

  Kernels::update_ell_diagonal_kernel<value_type, index_type, size_type>
      <<<NUM_BLOCKS, BLOCK_SIZE, 0>>>(A.nrows(), A.ncols(), num_entries_per_row,
                                      pitch, A.column_indices().data(),
                                      A.values().data(), diagonal.data());
#if defined(DEBUG) || defined(MORPHEUS_DEBUG)
  getLastCudaError("update_ell_diagonal_kernel: Kernel execution failed");
#endif
}

template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  throw Morpheus::NotImplementedException("get_diagonal not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename SizeType,
          typename ValueType>
void set_value(Matrix&, SizeType, SizeType, ValueType,
               typename std::enable_if_t<
                   Morpheus::is_ell_matrix_format_container_v<Matrix> &&
                   Morpheus::has_custom_backend_v<ExecSpace> &&
                   Morpheus::has_cuda_execution_space_v<ExecSpace> &&
                   Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  throw Morpheus::NotImplementedException("set_value not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
void set_values(
    Matrix&, typename IndexVector::value_type, const IndexVector,
    typename IndexVector::value_type, const IndexVector, const ValueVector,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<IndexVector> &&
        Morpheus::is_dense_vector_format_container_v<ValueVector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, IndexVector, ValueVector>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("set_values not implemented yet");
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix&, TransposeMatrix&,
    typename std::enable_if_t<
        Morpheus::is_ell_matrix_format_container_v<Matrix> &&
        Morpheus::is_ell_matrix_format_container_v<TransposeMatrix> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_cuda_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, TransposeMatrix>>* =
        nullptr) {
  throw Morpheus::NotImplementedException("transpose not implemented yet");
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_ELL_CUDA_MATRIXOPERATIONS_IMPL_HPP