/**
 * csr_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_CSR_MATRIX_HPP
#define MORPHEUS_CONTAINERS_CSR_MATRIX_HPP

#include <iostream>
#include <string>
#include <vector>

#include <morpheus/core/exceptions.hpp>
#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/matrix_tags.hpp>

namespace Morpheus {

struct CsrTag : public Impl::SparseMatTag {};

template <class... Properties>
class CsrMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type       = CsrMatrix<Properties...>;
  using traits     = Impl::MatrixTraits<Properties...>;
  using index_type = typename traits::index_type;
  using value_type = typename traits::value_type;
  using tag        = typename FormatTag<CsrTag>::tag;

  // Construct an empty CsrMatrix
  inline CsrMatrix() {}

  // Construct a CsrMatrix with a specific shape and number of non-zero entries
  inline CsrMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : _m(num_rows),
        _n(num_cols),
        _nnz(num_entries),
        _row_offsets(num_rows + 1),
        _column_indices(num_entries),
        _values(num_entries) {}

  // Construct from another matrix type
  template <typename MatrixType>
  CsrMatrix(const MatrixType &matrix) {
    // TODO: CsrMatrix(const MatrixType& matrix)
    Morpheus::NotImplementedException("CsrMatrix(const MatrixType& matrix)");
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    _m   = num_rows;
    _n   = num_cols;
    _nnz = num_entries;
    _row_offsets.resize(_m + 1);
    _column_indices.resize(_nnz);
    _values.resize(_nnz);
  }

  // Swap the contents of two CsrMatrix objects.
  void swap(CsrMatrix &matrix) {
    // TODO: swap(CsrMatrix& matrix)
    Morpheus::NotImplementedException(
        "CsrMatrix.swap(const CsrMatrix& matrix)");
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  CsrMatrix &operator=(const MatrixType &matrix) {
    // TODO: operator=(const MatrixType& matrix)
    Morpheus::NotImplementedException(
        "CsrMatrix.operator=(const MatrixType& matrix)");
  }

  // Accessors
  inline const index_type roff(const index_type idx) const {
    return _row_offsets[idx];
  }

  inline const index_type cind(const index_type idx) const {
    return _column_indices[idx];
  }

  inline value_type val(const index_type idx) const { return _values[idx]; }

  // Unified routines across all formats
  inline std::string name() const { return _name; }
  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _nnz; }

 private:
  std::string _name = "CsrMatrix";
  index_type _m, _n, _nnz;

  // TODO: Use Morpheus::array instead of std::vector
  using index_array_type = std::vector<index_type>;
  using value_array_type = std::vector<value_type>;

  index_array_type _row_offsets, _column_indices;
  value_array_type _values;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_CSR_MATRIX_HPP