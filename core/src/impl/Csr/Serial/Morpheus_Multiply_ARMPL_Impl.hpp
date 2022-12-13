/**
 * Morpheus_Multiply_ARMPL_Impl.hpp
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

#ifndef MORPHEUS_CSR_SERIAL_MULTIPLY_ARMPL_IMPL_HPP
#define MORPHEUS_CSR_SERIAL_MULTIPLY_ARMPL_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)
#if defined(MORPHEUS_ENABLE_TPL_ARMPL)
#include "armpl.h"

namespace Morpheus {
namespace Impl {

template <typename IndexType>
void multiply_armpl_csr(const size_t M, const size_t N, const IndexType* roff,
                        const IndexType* cind, const double* vals,
                        const double* x, double* y, bool init) {
  armpl_spmat_t armpl_mat;
  double beta = init ? 0.0 : 1.0;

  armpl_status_t info = armpl_spmat_create_csr_d(
      &armpl_mat, M, N, roff, cind, vals, ARMPL_SPARSE_CREATE_NOCOPY);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmat_create_coo_d returned %d\n", info);

  info = armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, 1.0, armpl_mat, x,
                           beta, y);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmv_exec_d returned %d\n", info);

  info = armpl_spmat_destroy(armpl_mat);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmat_destroy returned %d\n", info);
}

template <typename IndexType>
void multiply_armpl_csr(const size_t M, const size_t N, const IndexType* roff,
                        const IndexType* cind, const float* vals,
                        const float* x, float* y, bool init) {
  armpl_spmat_t armpl_mat;
  float beta = init ? 0.0 : 1.0;

  armpl_status_t info = armpl_spmat_create_csr_s(
      &armpl_mat, M, N, roff, cind, vals, ARMPL_SPARSE_CREATE_NOCOPY);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmat_create_coo_s returned %d\n", info);

  info = armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS, 1.0, armpl_mat, x,
                           beta, y);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmv_exec_s returned %d\n", info);

  info = armpl_spmat_destroy(armpl_mat);
  if (info != ARMPL_STATUS_SUCCESS)
    printf("ERROR: armpl_spmat_destroy returned %d\n", info);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_TPL_ARMPL
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_CSR_SERIAL_MULTIPLY_ARMPL_IMPL_HPP