/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_AlgorithmTags.hpp>

#include <impl/Morpheus_CudaUtils.hpp>
#include <impl/Coo/Kernels/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

// forward decl
template <typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void __spmv_coo_flat(const LinearOperator& A, const MatrixOrVector1& x,
                     MatrixOrVector2& y);

template <typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void __spmv_coo_serial(const LinearOperator& A, const MatrixOrVector1& x,
                       MatrixOrVector2& y);

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CooTag, DenseVectorTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  __spmv_coo_flat(A, x, y);
}

template <typename ExecSpace, typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
inline void multiply(
    const LinearOperator& A, const MatrixOrVector1& x, MatrixOrVector2& y,
    CooTag, DenseVectorTag, DenseVectorTag, Alg1,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space,
                               LinearOperator, MatrixOrVector1,
                               MatrixOrVector2>>* = nullptr) {
  __spmv_coo_serial(A, x, y);
}

template <typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void __spmv_coo_serial(const LinearOperator& A, const MatrixOrVector1& x,
                       MatrixOrVector2& y) {
  using IndexType    = typename LinearOperator::index_type;
  using ValueType    = typename LinearOperator::value_type;
  const IndexType* I = A.row_indices.data();
  const IndexType* J = A.column_indices.data();
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  Kernels::spmv_coo_serial_kernel<IndexType, ValueType>
      <<<1, 1>>>(A.nnnz(), I, J, V, x_ptr, y_ptr);
}

//////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
// Copyright 2008-2014 NVIDIA Corporation
// spmv_coo_flat
//   The input coo_matrix must be sorted by row.  Columns within each row
//   may appear in any order and duplicate entries are also acceptable.
//   This sorted COO format is easily obtained by expanding the row pointer
//   of a CSR matrix (csr.Ap) into proper row indices and then copying
//   the arrays containing the CSR column indices (csr.Aj) and nonzero values
//   (csr.Ax) verbatim.  A segmented reduction is used to compute the per-row
//   sums.
//
//
template <typename LinearOperator, typename MatrixOrVector1,
          typename MatrixOrVector2>
void __spmv_coo_flat(const LinearOperator& A, const MatrixOrVector1& x,
                     MatrixOrVector2& y) {
  using IndexType    = typename LinearOperator::index_type;
  using ValueType    = typename LinearOperator::value_type;
  const IndexType* I = A.row_indices.data();
  const IndexType* J = A.column_indices.data();
  const ValueType* V = A.values.data();

  const ValueType* x_ptr = x.data();
  ValueType* y_ptr       = y.data();

  if (A.nnnz() == 0) {
    // empty matrix
    return;
  } else if (A.nnnz() < static_cast<IndexType>(CUDA_WARP_SIZE)) {
    // small matrix
    Kernels::spmv_coo_serial_kernel<IndexType, ValueType>
        <<<1, 1, 0>>>(A.nnnz(), I, J, V, x_ptr, y_ptr);
    return;
  }

  const size_t BLOCK_SIZE = 256;
  const size_t MAX_BLOCKS = max_active_blocks(
      Kernels::spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE>,
      BLOCK_SIZE, (size_t)0);
  const size_t WARPS_PER_BLOCK = BLOCK_SIZE / CUDA_WARP_SIZE;

  const size_t num_units  = A.nnnz() / CUDA_WARP_SIZE;
  const size_t num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
  const size_t num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
  const size_t num_iters  = DIVIDE_INTO(num_units, num_warps);

  const IndexType interval_size = CUDA_WARP_SIZE * num_iters;

  const IndexType tail =
      num_units * CUDA_WARP_SIZE;  // do the last few nonzeros separately (fewer
                                   // than WARP_SIZE elements)

  const unsigned int active_warps =
      (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);

  Morpheus::DenseVector<IndexType, Kokkos::Cuda> temp_rows(active_warps);
  Morpheus::DenseVector<ValueType, Kokkos::Cuda> temp_vals(active_warps);
  IndexType* I_temp = A.column_indices.data();
  ValueType* V_temp = A.values.data();

  Kernels::spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE>
      <<<num_blocks, BLOCK_SIZE, 0>>>(tail, interval_size, I, J, V, x_ptr,
                                      y_ptr, I_temp, V_temp);

  Kernels::spmv_coo_reduce_update_kernel<IndexType, ValueType, BLOCK_SIZE>
      <<<1, BLOCK_SIZE, 0>>>(active_warps, I_temp, V_temp, y_ptr);

  Kernels::spmv_coo_serial_kernel<IndexType, ValueType><<<1, 1, 0>>>(
      A.nnnz() - tail, I + tail, J + tail, V + tail, x_ptr, y_ptr);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP