/**
 * Morpheus_MatrixOperations_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef MORPHEUS_HYB_HIP_MATRIXOPERATIONS_IMPL_HPP
#define MORPHEUS_HYB_HIP_MATRIXOPERATIONS_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Morpheus_HIPUtils.hpp>
#include <impl/Ell/HIP/Morpheus_MatrixOperations_Impl.hpp>
#include <impl/Coo/HIP/Morpheus_MatrixOperations_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
void update_diagonal(
    Matrix& A, const Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  Impl::update_diagonal<ExecSpace>(A.ell(), diagonal);
  Impl::update_diagonal<ExecSpace>(A.coo(), diagonal);
}

template <typename ExecSpace, typename Matrix, typename Vector>
void get_diagonal(
    const Matrix& A, Vector& diagonal,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  Impl::get_diagonal<ExecSpace>(A.cell(), diagonal);
  Impl::get_diagonal<ExecSpace>(A.ccoo(), diagonal);
}

template <typename ExecSpace, typename Matrix, typename SizeType,
          typename ValueType>
void set_value(Matrix& A, SizeType row, SizeType col, ValueType value,
               typename std::enable_if_t<
                   Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
                   Morpheus::has_custom_backend_v<ExecSpace> &&
                   Morpheus::has_hip_execution_space_v<ExecSpace> &&
                   Morpheus::has_access_v<ExecSpace, Matrix>>* = nullptr) {
  Impl::set_value<ExecSpace>(A.ell(), row, col, value);
  Impl::set_value<ExecSpace>(A.coo(), row, col, value);
}

template <typename ExecSpace, typename Matrix, typename IndexVector,
          typename ValueVector>
void set_values(
    Matrix& A, const typename IndexVector::value_type m, const IndexVector idxm,
    const typename IndexVector::value_type n, const IndexVector idxn,
    const ValueVector values,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<IndexVector> &&
        Morpheus::is_dense_vector_format_container_v<ValueVector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, IndexVector, ValueVector>>* =
        nullptr) {
  Impl::set_values<ExecSpace>(A.ell(), m, idxm, n, idxn, values);
  Impl::set_values<ExecSpace>(A.coo(), m, idxm, n, idxn, values);
}

template <typename ExecSpace, typename Matrix, typename TransposeMatrix>
void transpose(
    const Matrix& A, TransposeMatrix& At,
    typename std::enable_if_t<
        Morpheus::is_hyb_matrix_format_container_v<Matrix> &&
        Morpheus::is_hyb_matrix_format_container_v<TransposeMatrix> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_hip_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, TransposeMatrix>>* =
        nullptr) {
  Impl::transpose<ExecSpace>(A.cell(), At.ell());
  Impl::transpose<ExecSpace>(A.ccoo(), At.coo());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_HYB_HIP_MATRIXOPERATIONS_IMPL_HPP
