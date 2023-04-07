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

#ifndef MORPHEUS_DIA_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP
#define MORPHEUS_DIA_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)
#if defined(MORPHEUS_ENABLE_ARM_SVE)

#include <impl/Arm/Morpheus_ArmSVE_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply_arm_sve(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
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

  if (init) {
    y.assign(y.size(), 0);
  }

  index_type* Adoff = A.cdiagonal_offsets().data();
  value_type* Aval  = A.cvalues().data();
  value_type* xval  = x.data();
  value_type* yval  = y.data();

  uint64_t const vl       = vcnt<SZ>();
  svuint_t<SZ> const vidx = vindex<SZ>(0, (uint_t<SZ>)A.ndiags());

  for (index_type row = 0; row < (index_type)A.nrows(); row += vl) {
    // vtype_t<SZ> vsum = vdup<SZ>((value_type)0);
    // NOTE: Could be replaced with a ptrue + drain loop with a bit more effort
    vbool_t const pg = vwhilelt<SZ>((uint_t<SZ>)row, (uint_t<SZ>)A.nrows());
    vtype_t<SZ> vsum = init ? vdup<SZ>((value_type)0) : svld1(pg, yval + row);

    for (size_type n = 0; n < A.ndiags(); n++) {
      index_type const col = row + Adoff[n];

      vbool_t p1 = vwhilelt<SZ>(col, (index_type)0);
      vbool_t p2 = vwhilelt<SZ>(col, (index_type)A.ncols());
      vbool_t pm = svbic_z(pg, p2, p1);  // pm = p2 && !p1

      // NOTE1: This code generates `bic+ptest+b.eq`; the `ptest` could be
      // avoided with `bics+b.eq`. Tested on GCC 11.2.0.
      // NOTE2: This test (sans `bics`) seems to be marginally detrimental for
      // performance. Possibly the CPU is smart enough to detect ops with
      // predicate=0? Might be worth eliminating. Tested with atmosmodl.mtx.
      // if(svptest_any(pm, pm)) {
      vtype_t<SZ> vAval =
          svld1_gather_index(pm, Aval + row * A.ndiags() + n, vidx);
      vtype_t<SZ> vxval = svld1(pm, xval + col);
      vsum              = svmla_m(pm, vsum, vAval, vxval);
      //}
    }

    svst1(pg, yval + row, vsum);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_ARM_SVE
#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DIA_SERIAL_MULTIPLY_ARMSVE_IMPL_HPP