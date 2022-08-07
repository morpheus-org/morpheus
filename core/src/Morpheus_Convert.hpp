/**
 * Morpheus_Convert.hpp
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

#ifndef MORPHEUS_CONVERT_HPP
#define MORPHEUS_CONVERT_HPP

#include <Morpheus_DynamicMatrix.hpp>
#include <impl/Morpheus_Convert_Impl.hpp>
#include <impl/Dynamic/Morpheus_Convert_Impl.hpp>

namespace Morpheus {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst) {
  Impl::convert<ExecSpace>(src, dst);
}

template <typename ExecSpace, typename SourceType>
void convert(SourceType& src, const formats_e index) {
  Morpheus::CooMatrix<
      typename SourceType::value_type, typename SourceType::index_type,
      typename SourceType::array_layout, typename SourceType::execution_space,
      typename SourceType::memory_traits>
      temp;
  Impl::convert<ExecSpace>(src, temp);
  src.activate(index);
  Impl::convert<ExecSpace>(temp, src);
}

template <typename ExecSpace, typename SourceType>
void convert(SourceType& src, const int index) {
  Morpheus::convert<ExecSpace>(src, static_cast<formats_e>(index));
}
}  // namespace Morpheus

#endif  // MORPHEUS_CONVERT_HPP