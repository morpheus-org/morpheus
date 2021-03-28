/**
 * dynamic_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP
#define MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP

#include <string>
#include <variant>
#include <functional>

#include <morpheus/containers/impl/dynamic_matrix_impl.hpp>
#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/matrix_tags.hpp>

namespace Morpheus {

struct DynamicTag : public Impl::MatrixTag {};

/** @class DynamicMatrix
 * @brief Dynamic Matrix class that acts as a sum type of all the supporting
 * Matrix Storage Formats.
 *
 * Template argument options:
 *  - DynamicMatrix<ValueType>
 *  - DynamicMatrix<ValueType, IndexType>
 *  - DynamicMatrix<ValueType, IndexType, Space>
 *  - DynamicMatrix<ValueType, Space>
 */
template <class... Properties>
class DynamicMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type   = DynamicMatrix<Properties...>;
  using traits = Impl::MatrixTraits<Properties...>;

  using index_type = typename traits::index_type;
  using value_type = typename traits::value_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;
  using tag             = typename MatrixFormatTag<DynamicTag>::tag;

  using reference       = DynamicMatrix &;
  using const_reference = const DynamicMatrix &;

  inline DynamicMatrix() = default;

  template <typename Format>
  inline DynamicMatrix(const Format &mat) : _formats(mat) {}

  template <typename... Args>
  inline void resize(const index_type m, const index_type n,
                     const index_type nnz, Args &&...args) {
    return std::visit(
        std::bind(Impl::any_type_resize<Properties...>(), std::placeholders::_1,
                  m, n, nnz, std::forward<Args>(args)...),
        _formats);
  }

  inline std::string name() const { return _name; }

  inline index_type nrows() const {
    return std::visit(Impl::any_type_get_nrows(), _formats);
  }

  inline index_type ncols() const {
    return std::visit(Impl::any_type_get_ncols(), _formats);
  }

  inline index_type nnnz() const {
    return std::visit(Impl::any_type_get_nnnz(), _formats);
  }

  inline std::string active_name() const {
    return std::visit(Impl::any_type_get_name(), _formats);
  }

  inline int active_index() const { return _formats.index(); }

  inline void activate(const formats_e index) {
    constexpr int size =
        std::variant_size_v<typename MatrixFormats<Properties...>::variant>;
    const int idx = static_cast<int>(index);

    if (idx > size) {
      std::cout << "Warning: There are " << size
                << " available formats to switch to. "
                << "Selecting to switch to format with index " << idx
                << " will default to the format with index 0." << std::endl;
    }
    Impl::activate_impl<size, Properties...>::activate(_formats, idx);
  }

  // Enable switching through direct integer indexing
  inline void activate(const int index) {
    activate(static_cast<formats_e>(index));
  }

 private:
  std::string _name = "DynamicMatrix";
  typename MatrixFormats<Properties...>::variant _formats;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP