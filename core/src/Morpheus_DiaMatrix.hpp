/**
 * Morpheus_DiaMatrix.hpp
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

#ifndef MORPHEUS_DIAMATRIX_HPP
#define MORPHEUS_DIAMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_DynamicMatrix.hpp>

#include <impl/Morpheus_MatrixBase.hpp>
#include <impl/Morpheus_Utils.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class DiaMatrix : public Impl::MatrixBase<DiaMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<DiaMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<DiaMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<DiaTag>::tag;

  using value_type           = typename traits::value_type;
  using non_const_value_type = typename traits::non_const_value_type;
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
      Morpheus::DenseMatrix<value_type, index_type, array_layout,
                            execution_space, memory_traits>;
  using const_value_array_type = const value_array_type;
  using value_array_pointer    = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;
  using const_value_array_reference = const value_array_reference;

  ~DiaMatrix()                 = default;
  DiaMatrix(const DiaMatrix &) = default;
  DiaMatrix(DiaMatrix &&)      = default;
  DiaMatrix &operator=(const DiaMatrix &) = default;
  DiaMatrix &operator=(DiaMatrix &&) = default;

  // Construct an empty DiaMatrix
  inline DiaMatrix()
      : base(), _ndiags(0), _alignment(0), _diagonal_offsets(), _values() {}

  // Construct a DiaMatrix with:
  //      a specific shape
  //      number of non-zero entries
  //      number of occupied diagonals
  //      amount of padding used to align the data (default=32)
  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries, const index_type num_diagonals,
                   const index_type alignment = 32)
      : base(num_rows, num_cols, num_entries),
        _ndiags(num_diagonals),
        _alignment(alignment),
        _diagonal_offsets(num_diagonals) {
    _values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                   num_diagonals);
  }

  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries,
                   const index_array_type &diag_offsets,
                   const value_array_type &vals)
      : base(num_rows, num_cols, num_entries),
        _diagonal_offsets(diag_offsets),
        _values(vals) {
    _ndiags    = _diagonal_offsets.size();
    _alignment = _values.nrows();
  }

  template <typename ValuePtr, typename IndexPtr>
  explicit inline DiaMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexPtr diag_offsets_ptr,
      ValuePtr vals_ptr, const index_type num_diagonals,
      const index_type alignment = 32,
      typename std::enable_if<std::is_pointer<ValuePtr>::value &&
                              std::is_pointer<IndexPtr>::value>::type * =
          nullptr)
      : base(num_rows, num_cols, num_entries),
        _diagonal_offsets(num_diagonals, diag_offsets_ptr),
        _values(Impl::get_pad_size<index_type>(num_rows, alignment),
                num_diagonals, vals_ptr) {}

  // Construct from another matrix type (Shallow)
  template <class VR, class... PR>
  DiaMatrix(const DiaMatrix<VR, PR...> &src,
            typename std::enable_if<is_compatible_type<
                DiaMatrix, typename DiaMatrix<VR, PR...>::type>::value>::type
                * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _ndiags(src.ndiags()),
        _alignment(src.alignment()),
        _diagonal_offsets(src.cdiagonal_offsets()),
        _values(src.cvalues()) {}

  // Assignment from another matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<DiaMatrix, typename DiaMatrix<VR, PR...>::type>::value,
      DiaMatrix &>::type
  operator=(const DiaMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _ndiags           = src.ndiags();
    _alignment        = src.alignment();
    _diagonal_offsets = src.cdiagonal_offsets();
    _values           = src.cvalues();

    return *this;
  }

  // Construct from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  DiaMatrix(
      const DynamicMatrix<VR, PR...> &src,
      typename std::enable_if<is_dynamically_compatible<
          DiaMatrix, typename DynamicMatrix<VR, PR...>::type>::value>::type * =
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
      is_dynamically_compatible<DiaMatrix,
                                typename DynamicMatrix<VR, PR...>::type>::value,
      DiaMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  // Construct from another matrix type
  template <typename MatrixType>
  DiaMatrix(const MatrixType &src) = delete;

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals,
                     const index_type alignment = 32) {
    base::resize(num_rows, num_cols, num_entries);
    _ndiags    = num_diagonals;
    _alignment = alignment;

    if (this->exceeds_tolerance(num_rows, num_entries, _ndiags)) {
      throw Morpheus::FormatConversionException(
          "DiaMatrix fill-in would exceed maximum tolerance");
    }

    _diagonal_offsets.resize(num_diagonals);
    _values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                   num_diagonals);
  }

  template <class VR, class... PR>
  inline void resize(const DiaMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz(), src.ndiags(), src.alignment());
  }

  template <class VR, class... PR>
  inline DiaMatrix &allocate(const DiaMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz(), src.ndiags(), src.alignment());
    return *this;
  }

  formats_e format_enum() const { return _id; }

  int format_index() const { return static_cast<int>(_id); }

  bool exceeds_tolerance(const index_type num_rows,
                         const index_type num_entries,
                         const index_type num_diagonals) {
    const float max_fill   = 10.0;
    const float threshold  = 10e9;  // 100M entries
    const float size       = float(num_diagonals) * float(num_rows);
    const float fill_ratio = size / std::max(1.0f, float(num_entries));

    if (max_fill < fill_ratio && size > threshold) {
      return true;
    } else {
      return false;
    }
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  diagonal_offsets(index_type n) {
    return _diagonal_offsets(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(index_type i,
                                                             index_type j) {
    return _values(i, j);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  cdiagonal_offsets(index_type n) const {
    return _diagonal_offsets(n);
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_reference
  cvalues(index_type i, index_type j) const {
    return _values(i, j);
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &diagonal_offsets() {
    return _diagonal_offsets;
  }

  MORPHEUS_FORCEINLINE_FUNCTION value_array_type &values() { return _values; }

  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &cdiagonal_offsets()
      const {
    return _diagonal_offsets;
  }

  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_type &cvalues() const {
    return _values;
  }

  MORPHEUS_FORCEINLINE_FUNCTION index_type ndiags() const { return _ndiags; }
  MORPHEUS_FORCEINLINE_FUNCTION index_type alignment() const {
    return _alignment;
  }

  MORPHEUS_FORCEINLINE_FUNCTION void set_ndiags(
      const index_type num_diagonals) {
    _ndiags = num_diagonals;
  }
  MORPHEUS_FORCEINLINE_FUNCTION void set_alignment(const index_type alignment) {
    _alignment = alignment;
  }

 private:
  index_type _ndiags, _alignment;
  index_array_type _diagonal_offsets;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::DIA_FORMAT;
};

}  // namespace Morpheus

#endif  // MORPHEUS_DIAMATRIX_HPP