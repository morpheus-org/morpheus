/**
 * matrix_tags.hpp
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

#ifndef MORPHEUS_CORE_MATRIX_TAGS_HPP
#define MORPHEUS_CORE_MATRIX_TAGS_HPP

namespace Morpheus {

namespace Impl {
struct MatrixTag {};

struct SparseMatTag : public Impl::MatrixTag {};
struct DenseMatTag : public Impl::MatrixTag {};
}  // namespace Impl

// Format Tag Wrapper
template <class T>
struct FormatTag {
  static_assert(std::is_base_of<Impl::MatrixTag, T>::value,
                "Morpheus: Invalid tag.");
  using format_tag = FormatTag;
  using tag        = T;
};
}  // namespace Morpheus
#endif  // MORPHEUS_CORE_MATRIX_TAGS_HPP