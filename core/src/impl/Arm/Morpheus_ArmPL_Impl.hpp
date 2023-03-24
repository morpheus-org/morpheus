/**
 * Morpheus_ArmPL_Impl.hpp
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

#ifndef MORPHEUS_ARM_ARMPL_IMPL_HPP
#define MORPHEUS_ARM_ARMPL_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL) || defined(MORPHEUS_ENABLE_OPENMP)
#if defined(MORPHEUS_ENABLE_TPL_ARMPL)

#include "armpl.h"

template <typename T>
armpl_status_t armpl_spmat_create_csr(
    armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr,
    const armpl_int_t *col_indx, const T *vals, armpl_int_t flags,
    typename std::enable_if_t<
        std::is_same_v<double, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmat_create_csr_d(A, m, n, row_ptr, col_indx, vals, flags);
}

template <typename T>
armpl_status_t armpl_spmat_create_csr(
    armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr,
    const armpl_int_t *col_indx, const T *vals, armpl_int_t flags,
    typename std::enable_if_t<
        std::is_same_v<float, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmat_create_csr_s(A, m, n, row_ptr, col_indx, vals, flags);
}

template <typename T>
armpl_status_t armpl_spmat_create_coo(
    armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz,
    const armpl_int_t *row_indx, const armpl_int_t *col_indx, const T *vals,
    armpl_int_t flags,
    typename std::enable_if_t<
        std::is_same_v<double, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmat_create_coo_d(A, m, n, nnz, row_indx, col_indx, vals,
                                  flags);
}

template <typename T>
armpl_status_t armpl_spmat_create_coo(
    armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz,
    const armpl_int_t *row_indx, const armpl_int_t *col_indx, const T *vals,
    armpl_int_t flags,
    typename std::enable_if_t<
        std::is_same_v<float, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmat_create_coo_s(A, m, n, nnz, row_indx, col_indx, vals,
                                  flags);
}

template <typename T>
armpl_status_t armpl_spmv_exec(
    enum armpl_sparse_hint_value trans, T alpha, armpl_spmat_t A, const T *x,
    T beta, T *y,
    typename std::enable_if_t<
        std::is_same_v<double, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmv_exec_d(trans, alpha, A, x, beta, y);
}

template <typename T>
armpl_status_t armpl_spmv_exec(
    enum armpl_sparse_hint_value trans, T alpha, armpl_spmat_t A, const T *x,
    T beta, T *y,
    typename std::enable_if_t<
        std::is_same_v<float, typename std::remove_cv_t<T>>> * = nullptr) {
  return armpl_spmv_exec_s(trans, alpha, A, x, beta, y);
}

#endif  // MORPHEUS_ENABLE_SERIAL || MORPHEUS_ENABLE_OPENMP
#endif  // MORPHEUS_ENABLE_TPL_ARMPL
#endif  // MORPHEUS_ARM_ARMPL_IMPL_HPP