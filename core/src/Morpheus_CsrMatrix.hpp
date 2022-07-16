/**
 * Morpheus_CsrMatrix.hpp
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

#ifndef MORPHEUS_CSRMATRIX_HPP
#define MORPHEUS_CSRMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DynamicMatrix.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class CsrMatrix : public Impl::MatrixBase<CsrMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<CsrMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<CsrMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::CsrFormatTag>::tag;

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

  using index_array_type =
      Morpheus::DenseVector<index_type, index_type, array_layout,
                            execution_space, memory_traits>;
  using const_index_array_type = const index_array_type;
  using index_array_pointer    = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;
  using const_index_array_reference = const index_array_reference;

  using value_array_type =
      Morpheus::DenseVector<value_type, index_type, array_layout,
                            execution_space, memory_traits>;
  using const_value_array_type = const value_array_type;
  using value_array_pointer    = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;
  using const_value_array_reference = const value_array_reference;

  ~CsrMatrix()                 = default;
  CsrMatrix(const CsrMatrix &) = default;
  CsrMatrix(CsrMatrix &&)      = default;
  CsrMatrix &operator=(const CsrMatrix &) = default;
  CsrMatrix &operator=(CsrMatrix &&) = default;

  // Construct an empty CsrMatrix
  inline CsrMatrix()
      : base(), _row_offsets(1), _column_indices(0), _values(0) {}

  // Construct a CsrMatrix with a specific shape and number of non-zero entries
  inline CsrMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : base(num_rows, num_cols, num_entries),
        _row_offsets(num_rows + 1),
        _column_indices(num_entries),
        _values(num_entries) {}

  template <typename ValuePtr, typename IndexPtr>
  explicit inline CsrMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexPtr roff_ptr, IndexPtr cind_ptr,
      ValuePtr vals_ptr,
      typename std::enable_if<std::is_pointer<ValuePtr>::value &&
                              std::is_pointer<IndexPtr>::value>::type * =
          nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_offsets(num_rows + 1, roff_ptr),
        _column_indices(num_entries, cind_ptr),
        _values(num_entries, vals_ptr) {}

  // Construct from another matrix type (Shallow)
  // Needs to be a compatible type
  template <class VR, class... PR>
  CsrMatrix(const CsrMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                CsrMatrix, typename CsrMatrix<VR, PR...>::type>::value>::type
                * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _row_offsets(src.crow_offsets()),
        _column_indices(src.ccolumn_indices()),
        _values(src.cvalues()) {}

  // Assignment from another matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<CsrMatrix,
                           typename CsrMatrix<VR, PR...>::type>::value,
      CsrMatrix &>::type
  operator=(const CsrMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _row_offsets    = src.crow_offsets();
    _column_indices = src.ccolumn_indices();
    _values         = src.cvalues();

    return *this;
  }

  // Construct from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  CsrMatrix(
      const DynamicMatrix<VR, PR...> &src,
      typename std::enable_if<is_dynamically_compatible<
          CsrMatrix, typename DynamicMatrix<VR, PR...>::type>::value>::type * =
          nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  // Assignment from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  typename std::enable_if<
      is_dynamically_compatible<CsrMatrix,
                                typename DynamicMatrix<VR, PR...>::type>::value,
      CsrMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  // Construct from another matrix type
  template <typename MatrixType>
  CsrMatrix(const MatrixType &src) = delete;

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    _row_offsets.resize(num_rows + 1);
    _column_indices.resize(num_entries);
    _values.resize(num_entries);
  }

  template <class VR, class... PR>
  inline void resize(const CsrMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
  }

  template <class VR, class... PR>
  inline CsrMatrix &allocate(const CsrMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }

  formats_e format_enum() const { return _id; }

  int format_index() const { return static_cast<int>(_id); }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  row_offsets(index_type n) {
    return _row_offsets(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  column_indices(index_type n) {
    return _column_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(index_type n) {
    return _values(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  crow_offsets(index_type n) const {
    return _row_offsets(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  ccolumn_indices(index_type n) const {
    return _column_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_reference
  cvalues(index_type n) const {
    return _values(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &row_offsets() {
    return _row_offsets;
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &column_indices() {
    return _column_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_type &values() { return _values; }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &crow_offsets() const {
    return _row_offsets;
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &ccolumn_indices()
      const {
    return _column_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_type &cvalues() const {
    return _values;
  }

 private:
  index_array_type _row_offsets, _column_indices;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::CSR_FORMAT;
};

}  // namespace Morpheus

#endif  // MORPHEUS_CSRMATRIX_HPP