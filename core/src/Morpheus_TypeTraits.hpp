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

#include <Morpheus_FormatTags.hpp>
#include <fwd/Morpheus_Fwd_MatrixBase.hpp>
#include <fwd/Morpheus_Fwd_DenseMatrix.hpp>
#include <Kokkos_Core.hpp>

#include <type_traits>
#include <variant>

namespace Morpheus {

template <typename T, typename Variant>
struct is_variant_member;

template <typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>>
    : public std::disjunction<std::is_same<T, Ts>...> {};

template <typename T, typename Variant>
inline constexpr bool is_variant_member_v =
    is_variant_member<T, Variant>::value;

template <typename Tag>
struct is_matrix
    : std::integral_constant<bool,
                             std::is_base_of<Impl::MatrixTag, Tag>::value> {};

template <typename Tag>
inline constexpr bool is_matrix_v = is_matrix<Tag>::value;

template <typename Tag>
struct is_sparse_matrix
    : std::integral_constant<bool,
                             std::is_base_of<Impl::SparseMatTag, Tag>::value> {
};

template <template <class T, class... P> class Container, class T, class... P>
struct is_sparse_matrix_class
    : std::integral_constant<
          bool, std::is_base_of<Impl::MatrixBase<Container, T, P...>,
                                Container<T, P...>>::value> {};

template <template <class T, class... P> class Container, class T, class... P>
struct is_dense_matrix_container
    : std::integral_constant<
          bool, std::is_same<DenseMatrix<T, P...>, Container<T, P...>>::value> {
};

template <typename Tag>
inline constexpr bool is_sparse_matrix_v = is_sparse_matrix<Tag>::value;

template <typename Tag>
struct is_dense_matrix
    : std::integral_constant<bool,
                             std::is_base_of<Impl::DenseMatTag, Tag>::value> {};

template <typename Tag>
inline constexpr bool is_dense_matrix_v = is_dense_matrix<Tag>::value;

template <typename Tag>
struct is_vector
    : std::integral_constant<bool,
                             std::is_base_of<Impl::VectorTag, Tag>::value> {};

template <typename Tag>
inline constexpr bool is_vector_v = is_vector<Tag>::value;

template <typename T1, typename T2>
struct is_compatible_type
    : std::integral_constant<bool,
                             std::is_same<typename T1::memory_space,
                                          typename T2::memory_space>::value &&
                                 std::is_same<typename T1::value_type,
                                              typename T2::value_type>::value &&
                                 std::is_same<typename T1::index_type,
                                              typename T2::index_type>::value> {
};

template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class T>
struct is_arithmetic {
  static_assert(std::is_arithmetic<typename remove_cvref<T>::type>::value,
                "Morpheus: Invalid arithmetic type.");
  using container = is_arithmetic;
  using type      = T;
};

template <class ExecSpace, class T>
inline constexpr bool has_access_v =
    Kokkos::Impl::SpaceAccessibility<ExecSpace,
                                     typename T::memory_space>::accessible;

template <class ExecSpace>
inline constexpr bool is_execution_space_v =
    Kokkos::Impl::is_execution_space<ExecSpace>::value;

template <class Space>
inline constexpr bool is_Host_Memoryspace_v =
    std::is_same<Kokkos::HostSpace::memory_space,
                 typename Space::memory_space>::value;

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