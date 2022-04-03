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

#ifndef MORPHEUS_DYNAMIC_CONVERT_IMPL_HPP
#define MORPHEUS_DYNAMIC_CONVERT_IMPL_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
// forward decl
template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst);

namespace Impl {
struct convert_fn {
  using result_type = void;

  template <typename ExecSpace, typename SourceType, typename DestinationType>
  result_type operator()(const SourceType& src, DestinationType& dst) {
    Morpheus::convert<ExecSpace>(src, dst);
  }
};

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DynamicTag,
             SparseMatTag) {
  auto f = std::bind(Impl::convert_fn<ExecSpace>(), std::placeholders::_1,
                     std::ref(dst));
  Morpheus::Impl::Variant::visit(f, src.const_formats());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, SparseMatTag,
             DynamicTag) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  auto f = std::bind(Impl::convert_fn<ExecSpace>(), std::cref(src),
                     std::placeholders::_1);
  Morpheus::Impl::Variant::visit(f, dst.formats());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst, DynamicTag,
             DynamicTag) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  Morpheus::Impl::Variant::visit(Impl::convert_fn<ExecSpace>(),
                                 src.const_formats(), dst.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_CONVERT_IMPL_HPP