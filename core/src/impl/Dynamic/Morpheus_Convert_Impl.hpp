/**
 * Morpheus_Convert_Impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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
#include <impl/Morpheus_Convert_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace>
struct convert_fn {
  using result_type = void;

  template <typename SourceType, typename DestinationType>
  result_type operator()(const SourceType& src, DestinationType& dst) {
    Impl::convert<ExecSpace>(src, dst);
  }
};

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<SourceType>::value &&
        Morpheus::is_sparse_matrix_container<DestinationType>::value>::type* =
        nullptr) {
  auto f = std::bind(Impl::convert_fn<ExecSpace>(), std::placeholders::_1,
                     std::ref(dst));
  Impl::Variant::visit(f, src.const_formats());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst,
             typename std::enable_if<
                 Morpheus::is_sparse_matrix_container<SourceType>::value &&
                 Morpheus::is_dynamic_matrix_format_container<
                     DestinationType>::value>::type* = nullptr) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  auto f = std::bind(Impl::convert_fn<ExecSpace>(), std::cref(src),
                     std::placeholders::_1);
  Impl::Variant::visit(f, dst.formats());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_dynamic_matrix_format_container<SourceType>::value &&
        Morpheus::is_dynamic_matrix_format_container<DestinationType>::value>::
        type* = nullptr) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  Impl::Variant::visit(Impl::convert_fn<ExecSpace>(), src.const_formats(),
                       dst.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_CONVERT_IMPL_HPP