/**
 * Morpheus_Multiply_ArmSVE_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
 *
 * Contributing Authors:
 * Ricardo Jesus (rjj@ed.ac.uk)
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

#ifndef MORPHEUS_COO_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP
#define MORPHEUS_COO_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)
#if defined(MORPHEUS_ENABLE_ARM_SVE)

#include <impl/Arm/Morpheus_ArmSVE_Impl.hpp>
#include <impl/Morpheus_ArmUtils.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply_arm_sve(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using value_type = typename Matrix::value_type;
  using index_type = typename Matrix::index_type;
  using size_type  = typename Matrix::size_type;
  static_assert(sizeof(value_type) >= sizeof(index_type),
                "Unsafe index truncation.");
  const int SZ = sizeof(value_type) * 8;

  index_type* Arind = A.crow_indices().data();
  index_type* Acind = A.ccolumn_indices().data();
  value_type* Aval  = A.cvalues().data();
  value_type* xval  = x.data();
  value_type* yval  = y.data();

  if (init) {
    y.assign(y.size(), 0);
  }

  vbool_t pg;

  for (size_type n = 0; n < A.nnnz(); n += vcntp<SZ>(pg, pg)) {
    pg = vwhilelt<SZ>((uint_t<SZ>)n, (uint_t<SZ>)A.nnnz());

    auto vArind = vld1su<SZ>(pg, Arind + n);
    auto vAcind = vld1su<SZ>(pg, Acind + n);

    // NOTE: Can the indices in `Arind` be out of orther? If so, use the version
    // below. pg = svbrkb_z(pg, svcmpne(pg, vArind, Arind[n]));
    pg = svcmpeq(pg, vArind, Arind[n]);

    vtype_t<SZ> vAval = svld1(pg, Aval + n);
    vtype_t<SZ> vxval = svld1_gather_index(pg, xval, vAcind);
    vtype_t<SZ> vres  = svmul_x(pg, vAval, vxval);

    // NOTE: We do a tree-based reduction. If a left-to-right reduction is
    // preferred, use the version below.
    // yval[Arind[n]] = svadda(pg, yval[Arind[n]], vres);
    yval[Arind[n]] += svaddv(pg, vres);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_ARM_SVE
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_COO_SERIAL_MULTIPLY_ARMPL_IMPL_HPP