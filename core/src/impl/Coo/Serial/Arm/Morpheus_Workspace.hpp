/**
 * Morpheus_Workspace.hpp
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

#ifndef MORPHEUS_COO_ARM_WORKSPACE_HPP
#define MORPHEUS_COO_ARM_WORKSPACE_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)
#if defined(MORPHEUS_ENABLE_TPL_ARMPL)

#include <impl/Arm/Morpheus_ArmPL_Impl.hpp>
#include <impl/Morpheus_ArmUtils.hpp>
#include <unordered_map>

namespace Morpheus {
namespace Impl {

class ArmPLCooWorkspace_Serial {
 public:
  ArmPLCooWorkspace_Serial() : _handles_map(), _init(false) {}

  ~ArmPLCooWorkspace_Serial() {
    for (auto p = _handles_map.begin(); p != _handles_map.end(); p++)
      CHECK_ARMPL_ERROR(armpl_spmat_destroy(p->second));
  }

  template <typename ValueType>
  armpl_spmat_t handle(armpl_int_t m, armpl_int_t n, armpl_int_t nnz,
                       const armpl_int_t *row_indx, const armpl_int_t *col_indx,
                       const ValueType *vals, armpl_int_t flags) {
    armpl_int_t *key = (armpl_int_t *)row_indx;
    if (_handles_map.find(key) == _handles_map.end()) {
      armpl_spmat_t armpl_mat;
      armpl_spmat_create_coo<ValueType>(&armpl_mat, m, n, nnz, row_indx,
                                        col_indx, vals, flags);
      // Give hints and optimize spmv algorithm
      CHECK_ARMPL_ERROR(armpl_spmat_hint(armpl_mat,
                                         ARMPL_SPARSE_HINT_SPMV_OPERATION,
                                         ARMPL_SPARSE_OPERATION_NOTRANS));

      CHECK_ARMPL_ERROR(armpl_spmat_hint(armpl_mat,
                                         ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
                                         ARMPL_SPARSE_INVOCATIONS_MANY));
      CHECK_ARMPL_ERROR(armpl_spmv_optimize(armpl_mat));

      _handles_map[key] = armpl_mat;
    }
    return _handles_map[key];
  }

 private:
  std::unordered_map<armpl_int_t *, armpl_spmat_t> _handles_map;
  bool _init;
};

extern ArmPLCooWorkspace_Serial armplcoospace_serial;

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_TPL_ARMPL
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_COO_ARM_WORKSPACE_HPP