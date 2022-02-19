/**
 * Morpheus_CooMatrix.hpp
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

#ifndef MORPHEUS_COOMATRIX_HPP
#define MORPHEUS_COOMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DynamicMatrix.hpp>
#include <Morpheus_Sort.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class CooMatrix : public Impl::MatrixBase<CooMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<CooMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<CooMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::CooTag>::tag;

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

  ~CooMatrix()                 = default;
  CooMatrix(const CooMatrix &) = default;
  CooMatrix(CooMatrix &&)      = default;
  CooMatrix &operator=(const CooMatrix &) = default;
  CooMatrix &operator=(CooMatrix &&) = default;

  // Construct an empty CooMatrix
  inline CooMatrix()
      : base(), _row_indices(0), _column_indices(0), _values(0) {}

  // Construct a CooMatrix with a specific shape and number of non-zero entries
  inline CooMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : base(num_rows, num_cols, num_entries),
        _row_indices(num_entries),
        _column_indices(num_entries),
        _values(num_entries) {}

  template <typename ValuePtr, typename IndexPtr>
  explicit inline CooMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexPtr rind_ptr, IndexPtr cind_ptr,
      ValuePtr vals_ptr,
      typename std::enable_if<std::is_pointer<ValuePtr>::value &&
                              std::is_pointer<IndexPtr>::value>::type * =
          nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_indices(size_t(num_entries), rind_ptr),
        _column_indices(size_t(num_entries), cind_ptr),
        _values(size_t(num_entries), vals_ptr) {}

  // Construct from another matrix type (Shallow)
  // Needs to be a compatible type
  template <class VR, class... PR>
  CooMatrix(const CooMatrix<VR, PR...> &src,
            typename std::enable_if<is_compatible_type<
                CooMatrix, typename CooMatrix<VR, PR...>::type>::value>::type
                * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _row_indices(src.crow_indices()),
        _column_indices(src.ccolumn_indices()),
        _values(src.cvalues()) {}

  // Assignment from another matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<CooMatrix, typename CooMatrix<VR, PR...>::type>::value,
      CooMatrix &>::type
  operator=(const CooMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _row_indices    = src.crow_indices();
    _column_indices = src.ccolumn_indices();
    _values         = src.cvalues();

    return *this;
  }

  // Construct from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  CooMatrix(
      const DynamicMatrix<VR, PR...> &src,
      typename std::enable_if<is_compatible_container<
          CooMatrix, typename DynamicMatrix<VR, PR...>::type>::value>::type * =
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
      is_compatible_container<CooMatrix,
                              typename DynamicMatrix<VR, PR...>::type>::value,
      CooMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  // Construct from another matrix type
  template <typename MatrixType>
  CooMatrix(const MatrixType &src) = delete;

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    _row_indices.resize(num_entries);
    _column_indices.resize(num_entries);
    _values.resize(num_entries);
  }

  template <class VR, class... PR>
  inline void resize(const CooMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
  }

  template <class VR, class... PR>
  inline CooMatrix &allocate(const std::string name,
                             const CooMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }

  // Sort matrix elements by row index
  void sort_by_row(void) {
    // TODO: CooMatrix.sort_by_row
    throw Morpheus::NotImplementedException("CooMatrix.sort_by_row()");
  }

  // Sort matrix elements by row and column index
  void sort_by_row_and_column(void) { Morpheus::sort_by_row_and_column(*this); }

  // Determine whether matrix elements are sorted by row index
  bool is_sorted_by_row(void) {
    // TODO: CooMatrix.is_sorted_by_row
    throw Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row()");
    return true;
  }

  // Determine whether matrix elements are sorted by row and column index
  bool is_sorted(void) { return Morpheus::is_sorted(*this); }

  formats_e format_enum() const { return _id; }

  int format_index() const { return static_cast<int>(_id); }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  row_indices(index_type n) {
    return _row_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  column_indices(index_type n) {
    return _column_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(index_type n) {
    return _values(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  crow_indices(index_type n) const {
    return _row_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  ccolumn_indices(index_type n) const {
    return _column_indices(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_reference
  cvalues(index_type n) const {
    return _values(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &row_indices() {
    return _row_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &column_indices() {
    return _column_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_type &values() { return _values; }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &crow_indices() const {
    return _row_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &ccolumn_indices()
      const {
    return _column_indices;
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_type &cvalues() const {
    return _values;
  }

 private:
  index_array_type _row_indices, _column_indices;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::COO_FORMAT;
};
}  // namespace Morpheus

#endif  // MORPHEUS_COOMATRIX_HPP