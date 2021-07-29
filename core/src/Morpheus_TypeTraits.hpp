/**
 * Morpheus_TypeTraits.hpp
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
#ifndef MORPHEUS_TYPETRAITS_HPP
#define MORPHEUS_TYPETRAITS_HPP

#include <type_traits>

#include <Morpheus_Core.hpp>
#include <impl/Morpheus_MatrixTags.hpp>

namespace Morpheus {

namespace Impl {}

template <class T>
struct is_sparse_mat {
  using value =
      typename std::is_same<typename T::tag, Impl::SparseMatTag>::value;
};

template <class T>
using is_sparse_mat_v = typename is_sparse_mat<T>::value;

template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class ExecSpace, class T>
inline constexpr bool has_access_v =
    Kokkos::Impl::SpaceAccessibility<ExecSpace,
                                     typename T::memory_space>::accessible;

template <class ExecSpace>
inline constexpr bool is_execution_space_v =
    Kokkos::Impl::is_execution_space<ExecSpace>::value;

template <class ExecSpace>
inline constexpr bool is_Serial_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::Serial::execution_space>::value;

#if defined(MORPHEUS_ENABLE_OPENMP)
template <class ExecSpace>
inline constexpr bool is_OpenMP_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::OpenMP::execution_space>::value;
#endif  // MORPHEUS_ENABLE_OPENMP

#if defined(MORPHEUS_ENABLE_CUDA)
template <class ExecSpace>
inline constexpr bool is_Cuda_space_v =
    std::is_same<typename ExecSpace::execution_space,
                 Kokkos::Cuda::execution_space>::value;
#endif  // MORPHEUS_ENABLE_CUDA

}  // namespace Morpheus

#endif  // MORPHEUS_TYPETRAITS_HPP