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

#include <impl/Arm/Morpheus_ArmPL_Impl.hpp>
#include <impl/Morpheus_ArmUtils.hpp>
#include <impl/Csr/Serial/Arm/Morpheus_Workspace.hpp>

namespace Morpheus {
namespace Impl {

template <typename IndexType, typename ValueType>
void multiply_armpl_csr_serial(
    const size_t M, const size_t N, const IndexType* roff,
    const IndexType* cind, const ValueType* vals, const ValueType* x,
    ValueType* y, bool init,
    typename std::enable_if_t<std::is_floating_point_v<ValueType> &&
                              std::is_same_v<IndexType, int>>* = nullptr) {
  double beta             = init ? 0.0 : 1.0;
  armpl_spmat_t armpl_mat = armplcsrspace_serial.handle(
      M, N, roff, cind, vals, ARMPL_SPARSE_CREATE_NOCOPY);
  CHECK_ARMPL_ERROR(armpl_spmv_exec<ValueType>(ARMPL_SPARSE_OPERATION_NOTRANS,
                                               1.0, armpl_mat, x, beta, y));
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_TPL_ARMPL
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_CSR_SERIAL_MULTIPLY_ARMPL_IMPL_HPP