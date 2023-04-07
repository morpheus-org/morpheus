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

#ifndef MORPHEUS_CSR_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP
#define MORPHEUS_CSR_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP

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
        Morpheus::is_csr_matrix_format_container_v<Matrix> &&
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

  index_type* Aroff = A.crow_offsets().data();
  index_type* Acind = A.ccolumn_indices().data();
  value_type* Aval  = A.cvalues().data();
  value_type* xval  = x.data();
  value_type* yval  = y.data();

  if (init) {
    y.assign(y.size(), 0);
  }

  uint64_t const vl = vcnt<SZ>();

  // for (size_type i = 0; i < A.nrows(); i += vl) {
  //   vbool_t const pg = vwhilelt<SZ>((uint_t<SZ>)i, (uint_t<SZ>)A.nrows());
  //   svuint_t<SZ> const vidx_lim =
  //       vld1su<SZ>(pg, Aroff + i + 1);               // Aroff[i+1..]
  //   svuint_t<SZ> vidx = svinsr(vidx_lim, Aroff[i]);  // Aroff[i..]
  //   vtype_t<SZ> vsum  = vdup<SZ>((value_type)0);

  //   vbool_t pm = svcmplt(pg, vidx, vidx_lim);

  //   while (svptest_any(pm, pm)) {
  //     vtype_t<SZ> vAval = svld1_gather_index(pm, Aval, vidx);
  //     vtype_t<SZ> vxval =
  //         svld1_gather_index(pm, xval, svld1_gather_index(pm, Acind, vidx));

  //     vsum = svmla_m(pm, vsum, vAval, vxval);

  //     vidx = svadd_x(pm, vidx, 1);
  //     pm   = svcmplt(pm, vidx, vidx_lim);
  //   }

  //   svst1(pg, yval + i, vsum);
  // }
  for (size_type i = 0; i < A.nrows(); i++) {
    vtype_t<SZ> vsum = vdup<SZ>((value_type)0);
    // vtype_t<SZ> vsum = init ? vdup<SZ>((value_type)0) : svld1(pg, yval + i);
    for (index_type jj = Aroff[i]; jj < Aroff[i + 1]; jj += vl) {
      vbool_t const pg = vwhilelt<SZ>((uint_t<SZ>)jj, (uint_t<SZ>)Aroff[i + 1]);
      auto vAval       = svld1(pg, Aval + jj);
      auto vAcind      = vld1su<SZ>(pg, Acind + jj);
      auto vxval       = svld1_gather_index(pg, xval, vAcind);
      vsum             = svmla_m(pg, vsum, vAval, vxval);
    }
    yval[i] = init ? svaddv(vptrue<SZ>(), vsum)
                   : yval[i] + svaddv(vptrue<SZ>(), vsum);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_ARM_SVE
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_CSR_SERIAL_MULTIPLY_ARMPL_IMPL_HPP