/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_CONVERT_IMPL_HPP
#define MORPHEUS_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <fwd/Morpheus_Fwd_CooMatrix.hpp>

#include <impl/DenseVector/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/DenseVector/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/DenseVector/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/DenseVector/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/DenseVector/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/DenseMatrix/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/DenseMatrix/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/DenseMatrix/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/DenseMatrix/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/DenseMatrix/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Coo/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Csr/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Dia/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Dia/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Dia/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Dia/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Dia/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Ell/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Ell/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Ell/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Ell/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Ell/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Hyb/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Hyb/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Hyb/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Hyb/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Hyb/Kokkos/Morpheus_Convert_Impl.hpp>

#include <impl/Hdc/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Hdc/OpenMP/Morpheus_Convert_Impl.hpp>
#include <impl/Hdc/Cuda/Morpheus_Convert_Impl.hpp>
#include <impl/Hdc/HIP/Morpheus_Convert_Impl.hpp>
#include <impl/Hdc/Kokkos/Morpheus_Convert_Impl.hpp>

namespace Morpheus {
namespace Impl {

// convert src -> coo_matrix -> dst
template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst,
             typename std::enable_if_t<
                 is_matrix_container_v<SourceType> &&
                 is_matrix_container_v<DestinationType> &&
                 !is_dynamic_matrix_format_container_v<SourceType> &&
                 !is_dynamic_matrix_format_container_v<DestinationType> &&
                 !is_coo_matrix_format_container_v<SourceType> &&
                 !is_coo_matrix_format_container_v<DestinationType> &&
                 !has_same_format_v<SourceType, DestinationType>>* = nullptr) {
  using ValueType   = typename SourceType::value_type;
  using IndexType   = typename SourceType::index_type;
  using ArrayLayout = typename SourceType::array_layout;
  using MemorySpace = typename SourceType::memory_space;
  using Backend     = typename ExecSpace::backend;

  using Coo = CooMatrix<ValueType, IndexType, ArrayLayout,
                        Morpheus::Device<ExecSpace, MemorySpace, Backend>>;
  Coo tmp;

  Impl::convert<ExecSpace>(src, tmp);
  Impl::convert<ExecSpace>(tmp, dst);
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CONVERT_IMPL_HPP