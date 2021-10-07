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
template <typename Matrix, typename Vector>
void __spmv_coo_flat(const Matrix& A, const Vector& x, Vector& y);

template <typename Matrix, typename Vector>
void __spmv_coo_serial(const Matrix& A, const Vector& x, Vector& y);

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, CooTag, DenseVectorTag, Alg0,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  __spmv_coo_flat(A, x, y);
}

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, CooTag, DenseVectorTag, Alg1,
    typename std::enable_if_t<
        !Morpheus::is_kokkos_space_v<ExecSpace> &&
        Morpheus::is_Cuda_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  __spmv_coo_serial(A, x, y);
}

template <typename Matrix, typename Vector>
void __spmv_coo_serial(const Matrix& A, const Vector& x, Vector& y) {
  using IndexType    = typename Matrix::index_type;
  using ValueType    = typename Matrix::value_type;
  const IndexType* I = A.crow_indices().data();
  const IndexType* J = A.ccolumn_indices().data();
  const ValueType* V = A.cvalues().data();

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
template <typename Matrix, typename Vector>
void __spmv_coo_flat(const Matrix& A, const Vector& x, Vector& y) {
  using IndexType = typename Matrix::index_type;
  using ValueType = typename Matrix::value_type;

  y.assign(y.size(), 0);
  if (A.nnnz() == 0) {
    // empty matrix
    return;
  } else if (A.nnnz() < static_cast<IndexType>(CUDA_WARP_SIZE)) {
    // small matrix
    Kernels::spmv_coo_serial_kernel<IndexType, ValueType><<<1, 1, 0>>>(
        A.nnnz(), A.crow_indices().data(), A.ccolumn_indices().data(),
        A.cvalues().data(), x.data(), y.data());
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

  typename Matrix::index_array_type temp_rows(active_warps, 0);
  typename Matrix::value_array_type temp_vals(active_warps, 0);

  Kernels::spmv_coo_flat_kernel<IndexType, ValueType, BLOCK_SIZE>
      <<<num_blocks, BLOCK_SIZE, 0>>>(
          tail, interval_size, A.crow_indices().data(),
          A.ccolumn_indices().data(), A.cvalues().data(), x.data(), y.data(),
          temp_rows.data(), temp_vals.data());

  Kernels::spmv_coo_reduce_update_kernel<IndexType, ValueType, BLOCK_SIZE>
      <<<1, BLOCK_SIZE, 0>>>(active_warps, temp_rows.data(), temp_vals.data(),
                             y.data());

  Kernels::spmv_coo_serial_kernel<IndexType, ValueType>
      <<<1, 1, 0>>>(A.nnnz() - tail, A.crow_indices().data() + tail,
                    A.ccolumn_indices().data() + tail,
                    A.cvalues().data() + tail, x.data(), y.data());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_COO_CUDA_MULTIPLY_IMPL_HPP