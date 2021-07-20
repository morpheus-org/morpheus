/**
 * Morpheus_DiaMatrix.hpp
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

#ifndef MORPHEUS_DIAMATRIX_HPP
#define MORPHEUS_DIAMATRIX_HPP

#include <string>

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_Copy.hpp>

#include <impl/Morpheus_MatrixTraits.hpp>
#include <impl/Morpheus_FormatTags.hpp>

namespace Morpheus {

template <class... Properties>
class DiaMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type   = DiaMatrix<Properties...>;
  using traits = Impl::MatrixTraits<Properties...>;
  using tag    = typename MatrixFormatTag<DiaTag>::tag;

  using value_type = typename traits::value_type;
  using index_type = typename traits::index_type;
  using size_type  = typename traits::index_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using pointer         = DiaMatrix *;
  using const_pointer   = const DiaMatrix *;
  using reference       = DiaMatrix &;
  using const_reference = const DiaMatrix &;

  using index_array_type = Morpheus::vector<index_type, device_type>;
  using value_array_type =
      Morpheus::DenseMatrix<value_type, Kokkos::LayoutLeft, device_type>;
  using value_array_pointer   = typename value_array_type::pointer;
  using value_array_reference = typename value_array_type::reference;

  index_array_type diagonal_offsets;
  value_array_type values;

  ~DiaMatrix()                 = default;
  DiaMatrix(const DiaMatrix &) = default;
  DiaMatrix(DiaMatrix &&)      = default;
  reference operator=(const DiaMatrix &) = default;
  reference operator=(DiaMatrix &&) = default;

  // Construct an empty DiaMatrix
  inline DiaMatrix()
      : diagonal_offsets(),
        values(),
        _name("DiaMatrix"),
        _m(0),
        _n(0),
        _nnz(0) {}

  // Construct a DiaMatrix with:
  //      a specific shape
  //      number of non-zero entries
  //      number of occupied diagonals
  //      amount of padding used to align the data (default=32)
  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries, const index_type num_diagonals,
                   const index_type alignment = 32)
      : diagonal_offsets(num_diagonals),
        _name("DiaMatrix"),
        _m(num_rows),
        _n(num_cols),
        _nnz(num_entries) {
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

  inline DiaMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries,
                   const index_type num_diagonals,
                   const index_type alignment = 32)
      : diagonal_offsets(num_diagonals),
        _name(name + "(DiaMatrix)"),
        _m(num_rows),
        _n(num_cols),
        _nnz(num_entries) {
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

  // Construct from another matrix type
  template <typename MatrixType>
  DiaMatrix(const MatrixType &matrix) : _name("DiaMatrix") {
    Morpheus::copy(matrix, *this);
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals,
                     const index_type alignment = 32) {
    _m   = num_rows;
    _n   = num_cols;
    _nnz = num_entries;
    diagonal_offsets.resize(num_diagonals);
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
    return *this;
  }

  // Unified routines across all formats
  inline std::string name() const { return _name; }
  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _nnz; }
  inline void set_nrows(const index_type rows) { _m = rows; }
  inline void set_ncols(const index_type cols) { _n = cols; }
  inline void set_nnnz(const index_type nnz) { _nnz = nnz; }

 private:
  // Calculates padding to align the data based on the current diagonal length
  inline const index_type _pad_size(index_type diag_len, index_type alignment) {
    return alignment * ((diag_len + alignment - 1) / alignment);
  }

  std::string _name;
  index_type _m, _n, _nnz;
};
}  // namespace Morpheus

#endif  // MORPHEUS_DIAMATRIX_HPP