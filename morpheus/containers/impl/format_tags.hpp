/**
 * format_tags.hpp
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

#ifndef MORPHEUS_CONTAINERS_IMPL_FORMAT_TAGS_HPP
#define MORPHEUS_CONTAINERS_IMPL_FORMAT_TAGS_HPP

#include <morpheus/core/matrix_tags.hpp>
#include <morpheus/core/vector_tags.hpp>

namespace Morpheus {

struct CooTag : public Impl::SparseMatTag {};
struct CsrTag : public Impl::SparseMatTag {};
struct DiaTag : public Impl::SparseMatTag {};
struct DynamicTag : public Impl::SparseMatTag {};

struct DenseMatrixTag : public Impl::DenseMatTag {};
struct DenseVectorTag : public Impl::VectorTag {};

}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_IMPL_FORMAT_TAGS_HPP