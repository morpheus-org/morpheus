/**
 * Morpheus_DynamicMatrix.hpp
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

#ifndef MORPHEUS_DYNAMICMATRIX_HPP
#define MORPHEUS_DYNAMICMATRIX_HPP

#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_Variant.hpp>
#include <impl/Morpheus_MatrixBase.hpp>
#include <impl/Morpheus_DynamicMatrix_Impl.hpp>

#include <iostream>
#include <string>
#include <functional>

namespace Morpheus {

template <class ValueType, class... Properties>
class DynamicMatrix
    : public Impl::MatrixBase<DynamicMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<DynamicMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<DynamicMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::DynamicTag>::tag;

  using value_type           = typename traits::value_type;
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::index_type;
  using index_type           = typename traits::index_type;
  using non_const_index_type = typename traits::non_const_index_type;

  using array_layout    = typename traits::array_layout;
  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;
  using memory_traits   = typename traits::memory_traits;
  using HostMirror      = typename traits::HostMirror;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using variant_type =
      typename MatrixFormats<ValueType, Properties...>::variant;

  ~DynamicMatrix()                     = default;
  DynamicMatrix(const DynamicMatrix &) = default;
  DynamicMatrix(DynamicMatrix &&)      = default;
  DynamicMatrix &operator=(const DynamicMatrix &) = default;
  DynamicMatrix &operator=(DynamicMatrix &&) = default;

  // Default Constructor
  inline DynamicMatrix() : _formats() {}

  // Construct dynamic matrix from another concrete matrix format
  template <typename Matrix>
  inline DynamicMatrix(
      const Matrix &src,
      typename std::enable_if<
          is_variant_member_v<typename Matrix::type, variant_type>>::type * =
          nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    this->activate(src.format_enum());
    auto f = std::bind(Impl::any_type_assign(), std::cref(src),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
  }

  // Assignment from another matrix type
  template <typename Matrix>
  typename std::enable_if<
      is_variant_member_v<typename Matrix::type, variant_type>,
      DynamicMatrix &>::type
  operator=(const Matrix &matrix) {
    base::resize(matrix.nrows(), matrix.ncols(), matrix.nnnz());
    this->activate(matrix.format_enum());

    auto f = std::bind(Impl::any_type_assign(), std::cref(matrix),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
    return *this;
  }

  // Construct from another compatible dynamic matrix type
  template <class VR, class... PR>
  DynamicMatrix(
      const DynamicMatrix<VR, PR...> &src,
      typename std::enable_if<is_compatible_type<
          DynamicMatrix, typename DynamicMatrix<VR, PR...>::type>::value>::type
          * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    this->activate(src.active_index());  // switch to src format
    Morpheus::Impl::Variant::visit(Impl::any_type_assign(), src.const_formats(),
                                   _formats);
  }

  // Assignment from another compatible dynamic matrix type
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<DynamicMatrix,
                         typename DynamicMatrix<VR, PR...>::type>::value,
      DynamicMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    this->activate(src.active_index());  // switch to src format
    Morpheus::Impl::Variant::visit(Impl::any_type_assign(), src.const_formats(),
                                   _formats);

    return *this;
  }

  template <typename... Args>
  inline void resize(const index_type m, const index_type n,
                     const index_type nnz, Args &&...args) {
    base::resize(m, n, nnz);
    auto f = std::bind(Impl::any_type_resize<ValueType, Properties...>(),
                       std::placeholders::_1, m, n, nnz,
                       std::forward<Args>(args)...);
    return Morpheus::Impl::Variant::visit(f, _formats);
  }

  // Resize from a compatible dynamic matrix
  template <class VR, class... PR>
  inline void resize(
      const DynamicMatrix<VR, PR...> &src) {
    Morpheus::Impl::Variant::visit(Impl::any_type_resize_from_mat(),
                                   src.const_formats(), _formats);
  }

  // Resize from a member matrix format
  template <typename Matrix>
  inline void resize(
      const Matrix &src,
      typename std::enable_if<
          is_variant_member_v<typename Matrix::type, variant_type>>::type * =
          nullptr) {
    base::resize(src.nrows(), src.ncols(), src.nnnz());
    this->activate(src.format_enum());

    auto f = std::bind(Impl::any_type_resize_from_mat(), std::cref(src),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
  }

  template <class VR, class... PR>
  inline DynamicMatrix &allocate(const DynamicMatrix<VR, PR...> &src) {
    base::resize(src.nrows(), src.ncols(), src.nnnz());
    this->activate(src.active_index());  // switch to src format
    Morpheus::Impl::Variant::visit(Impl::any_type_allocate(),
                                   src.const_formats(), _formats);
    return *this;
  }

  inline int active_index() const { return _formats.index(); }

  int format_index() const { return this->active_index(); }

  inline formats_e active_enum() const {
    return static_cast<formats_e>(_formats.index());
  }

  inline formats_e format_enum() const { return this->active_enum(); }

  inline void activate(const formats_e index) {
    constexpr int size = Morpheus::Impl::Variant::variant_size_v<
        typename MatrixFormats<ValueType, Properties...>::variant>;
    const int idx = static_cast<int>(index);

    if (idx > size) {
      std::cout << "Warning: There are " << size
                << " available formats to switch to. "
                << "Selecting to switch to format with index " << idx
                << " will default to the format with index 0." << std::endl;
    }
    Impl::activate_impl<size, ValueType, Properties...>::activate(_formats,
                                                                  idx);
  }

  // Enable switching through direct integer indexing
  inline void activate(const int index) {
    activate(static_cast<formats_e>(index));
  }

  inline const variant_type &const_formats() const { return _formats; }
  inline variant_type &formats() { return _formats; }

 private:
  variant_type _formats;
};
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMICMATRIX_HPP
