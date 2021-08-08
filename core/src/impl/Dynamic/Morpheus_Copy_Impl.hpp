/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_COPY_IMPL_HPP
#define MORPHEUS_DYNAMIC_COPY_IMPL_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <variant>

namespace Morpheus {

// forward decl
template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst);

namespace Impl {
struct copy_fn {
  using result_type = void;

  template <typename SourceType, typename DestinationType>
  result_type operator()(
      const SourceType& src, DestinationType& dst,
      typename std::enable_if<
          std::is_same<typename SourceType::tag,
                       typename DestinationType::tag>::value>::type* =
          nullptr) {
    Morpheus::copy(src, dst);
  }

  // Needed for the compiler to generate all possible combinations
  template <typename SourceType, typename DestinationType>
  result_type operator()(
      const SourceType& src, DestinationType& dst,
      typename std::enable_if<
          !std::is_same<typename SourceType::tag,
                        typename DestinationType::tag>::value>::type* =
          nullptr) {
    throw Morpheus::FormatConversionException(
        "Morpheus::copy() is only available between the same container types. "
        "Please use Morpheus::convert() instead to perform conversions between "
        "different types.");
  }
};

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DynamicTag,
          SparseMatTag) {
  auto f = std::bind(Impl::copy_fn(), std::placeholders::_1, std::ref(dst));
  std::visit(f, src.formats());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, SparseMatTag,
          DynamicTag) {
  auto f = std::bind(Impl::copy_fn(), std::cref(src), std::placeholders::_1);
  std::visit(f, dst.formats());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DynamicTag, DynamicTag) {
  std::visit(Impl::copy_fn(), src.formats(), dst.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_COPY_IMPL_HPP