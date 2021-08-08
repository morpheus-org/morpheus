/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_CONVERT_IMPL_HPP
#define MORPHEUS_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <fwd/Morpheus_Fwd_CooMatrix.hpp>

#include <impl/DenseVector/Morpheus_Convert_Impl.hpp>
#include <impl/DenseMatrix/Morpheus_Convert_Impl.hpp>

// TODO: Let Cmake generate these
#include <impl/Coo/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/Morpheus_Convert_Impl.hpp>
#include <impl/Dia/Morpheus_Convert_Impl.hpp>

#include <impl/Dynamic/Morpheus_Convert_Impl.hpp>

namespace Morpheus {
namespace Impl {

// convert src -> coo_matrix -> dst
template <typename SourceType, typename DestinationType, typename Format1,
          typename Format2>
void convert(const SourceType& src, DestinationType& dst, Format1, Format2,
             typename std::enable_if_t<!std::is_same_v<Format1, DynamicTag> &&
                                       !std::is_same_v<Format2, DynamicTag>>* =
                 nullptr) {
  using ValueType   = typename SourceType::value_type;
  using IndexType   = typename SourceType::index_type;
  using ArrayLayout = typename SourceType::array_layout;
  using MemorySpace = typename SourceType::memory_space;

  using Coo = CooMatrix<ValueType, IndexType, ArrayLayout, MemorySpace>;
  Coo tmp;

  Morpheus::Impl::convert(src, tmp, Format1(), typename Coo::tag());
  Morpheus::Impl::convert(tmp, dst, typename Coo::tag(), Format2());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CONVERT_IMPL_HPP