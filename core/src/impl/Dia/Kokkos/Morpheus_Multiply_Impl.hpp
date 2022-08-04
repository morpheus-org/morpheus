/**
 * Morpheus_Multiply_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_GenericSpace.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_dia_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::is_generic_space_v<ExecSpace> &&
        Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
                               Vector>>* = nullptr) {
  using execution_space  = typename ExecSpace::execution_space;
  using value_array_type = typename Matrix::value_array_type::value_array_type;
  using index_array_type = typename Matrix::index_array_type::value_array_type;
  using index_type       = typename Matrix::index_type;
  using value_type       = typename Matrix::value_type;
  using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

  const value_array_type values                  = A.cvalues().const_view();
  const typename Vector::value_array_type x_view = x.const_view();
  const index_array_type diagonal_offsets = A.cdiagonal_offsets().const_view();
  typename Vector2::value_array_type y_view = y.view();
  index_type ndiag = A.cvalues().ncols(), ncols = A.ncols();

  const Kokkos::TeamPolicy<execution_space> policy(A.nrows(), Kokkos::AUTO,
                                                   Kokkos::AUTO);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const member_type& team_member) {
        const index_type row = team_member.league_rank();

        value_type sum = init ? value_type(0) : y_view[row];
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, ndiag),
            [=](const int& n, value_type& lsum) {
              const index_type col = row + diagonal_offsets[n];

              if (col >= 0 && col < ncols) {
                lsum += values(row, n) * x_view[col];
              }
            },
            sum);
        y_view[row] = sum;
      });
}

// template <typename ExecSpace, typename Matrix, typename Vector>
// inline void multiply(
//     const Matrix& A, const Vector& x, Vector& y,
//     typename std::enable_if_t<
//         Morpheus::is_dia_matrix_format_container_v<Matrix> &&
//         Morpheus::is_dense_vector_format_container_v<Vector> &&
//         Morpheus::is_generic_space_v<ExecSpace> &&
//         Morpheus::has_access_v<typename ExecSpace::execution_space, Matrix,
//                                Vector>>* = nullptr) {
//   using execution_space  = typename ExecSpace::execution_space;
//   using value_array_type = typename
//   Matrix::value_array_type::value_array_type; using index_array_type =
//   typename Matrix::index_array_type::value_array_type; using index_type =
//   typename Matrix::index_type; using value_type       = typename
//   Matrix::value_type; using range_policy =
//       Kokkos::RangePolicy<Kokkos::IndexType<index_type>, execution_space>;

//   const value_array_type values                   = A.cvalues().const_view();
//   const typename Vector::value_array_type x_view = x.const_view();
//   const index_array_type diagonal_offsets =
//   A.cdiagonal_offsets().const_view(); typename Vector2::value_array_type
//   y_view = y.view(); index_type ndiag = A.cvalues().ncols(), ncols =
//   A.ncols();

//   range_policy policy(0, A.nrows());

//   Kokkos::parallel_for(
//       policy, KOKKOS_LAMBDA(const index_type row) {
//         value_type sum = y_view[row];

//         for (index_type n = 0; n < ndiag; n++) {
//           const index_type col = row + diagonal_offsets[n];

//           if (col >= 0 && col < ncols) {
//             sum += values(row, n) * x_view[col];
//           }
//         }

//         y_view[row] = sum;
//       });
// }

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DIA_KOKKOS_MULTIPLY_IMPL_HPP