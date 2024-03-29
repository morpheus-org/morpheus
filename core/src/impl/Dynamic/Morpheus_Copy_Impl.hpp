/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_DYNAMIC_COPY_IMPL_HPP
#define MORPHEUS_DYNAMIC_COPY_IMPL_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Copy_Impl.hpp>
#include <impl/Morpheus_Variant.hpp>

namespace Morpheus {
namespace Impl {

struct copy_fn {
  using result_type = void;

  template <typename SourceType, typename DestinationType>
  result_type operator()(
      const SourceType& src, DestinationType& dst,
      typename std::enable_if_t<
          is_format_compatible<SourceType, DestinationType>::value ||
          is_format_compatible_different_space<
              SourceType, DestinationType>::value>* = nullptr) {
    Impl::copy(src, dst);
  }

  // Needed for the compiler to generate all possible combinations
  template <typename SourceType, typename DestinationType>
  result_type operator()(
      const SourceType& src, DestinationType& dst,
      typename std::enable_if_t<
          !(is_format_compatible<SourceType, DestinationType>::value ||
            is_format_compatible_different_space<
                SourceType, DestinationType>::value)>* = nullptr) {
    throw Morpheus::FormatConversionException(
        "Morpheus::copy() is only available between the same container types "
        "(" +
        std::to_string(src.format_index()) +
        " != " + std::to_string(dst.format_index()) +
        "). Please use Morpheus::convert() instead to perform conversions "
        "between different types.");
  }
};

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          typename std::enable_if_t<
              Morpheus::is_dynamic_matrix_format_container<SourceType>::value &&
              Morpheus::is_sparse_matrix_container<DestinationType>::value>* =
              nullptr) {
  if (src.const_formats().index() == static_cast<int>(dst.format_enum())) {
    auto f = std::bind(Impl::copy_fn(), std::placeholders::_1, std::ref(dst));
    Morpheus::Impl::Variant::visit(f, src.const_formats());
  } else {
    throw Morpheus::FormatConversionException(
        "Morpheus::copy() is only available between the same container types. "
        "Active type of dynamic matrix should match the type of destination "
        "matrix. Please use Morpheus::convert() instead to perform conversions "
        "between different types.");
  }
}

template <typename SourceType, typename DestinationType>
void copy(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if_t<
        Morpheus::is_sparse_matrix_container<SourceType>::value &&
        Morpheus::is_dynamic_matrix_format_container<DestinationType>::value>* =
        nullptr) {
  auto f = std::bind(Impl::copy_fn(), std::cref(src), std::placeholders::_1);
  Impl::Variant::visit(f, dst.formats());
}

template <typename SourceType, typename DestinationType>
void copy(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if_t<
        Morpheus::is_dynamic_matrix_format_container<SourceType>::value &&
        Morpheus::is_dynamic_matrix_format_container<DestinationType>::value>* =
        nullptr) {
  Impl::Variant::visit(Impl::copy_fn(), src.const_formats(), dst.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMIC_COPY_IMPL_HPP