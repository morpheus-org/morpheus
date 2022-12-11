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

#ifndef MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP
#define MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Coo/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/HIP/Morpheus_Multiply_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Matrix, typename Vector>
inline void multiply(
    const Matrix& A, const Vector& x, Vector& y, const bool init,
    typename std::enable_if_t<
        Morpheus::is_coo_matrix_format_container_v<Matrix> &&
        Morpheus::is_dense_vector_format_container_v<Vector> &&
        Morpheus::has_generic_backend_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Matrix, Vector>>* = nullptr) {
  using backend = Morpheus::CustomBackend<typename ExecSpace::execution_space>;
  Impl::multiply<backend>(A, x, y, init);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_COO_KOKKOS_MULTIPLY_IMPL_HPP
