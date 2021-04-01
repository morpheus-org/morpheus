/**
 * coo_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_COO_MATRIX_HPP
#define MORPHEUS_CONTAINERS_COO_MATRIX_HPP

#include <iostream>
#include <string>
#include <vector>

#include <morpheus/core/exceptions.hpp>
#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/matrix_tags.hpp>
#include <morpheus/containers/vector.hpp>

namespace Morpheus {

struct CooTag : public Impl::SparseMatTag {};

template <class... Properties>
class CooMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type   = CooMatrix<Properties...>;
  using traits = Impl::MatrixTraits<Properties...>;

  using index_type = typename traits::index_type;
  using value_type = typename traits::value_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;
  using tag             = typename MatrixFormatTag<CooTag>::tag;

  using index_array_type = Morpheus::vector<index_type, device_type>;
  using value_array_type = Morpheus::vector<value_type, device_type>;

  index_array_type row_indices, column_indices;
  value_array_type values;

  // Construct an empty CooMatrix
  inline CooMatrix()
      : row_indices(0),
        column_indices(0),
        values(0),
        _name("CooMatrix"),
        _m(0),
        _n(0),
        _nnz(0) {}

  // Construct a CooMatrix with a specific shape and number of non-zero entries
  inline CooMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : row_indices(num_entries),
        column_indices(num_entries),
        values(num_entries),
        _name("CooMatrix"),
        _m(num_rows),
        _n(num_cols),
        _nnz(num_entries) {}

  inline CooMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries)
      : row_indices(num_entries),
        column_indices(num_entries),
        values(num_entries),
        _name(name),
        _m(num_rows),
        _n(num_cols),
        _nnz(num_entries) {}

  // Construct from another matrix type
  template <typename MatrixType>
  CooMatrix(const MatrixType &matrix) {
    // TODO: CooMatrix(const MatrixType& matrix)
    Morpheus::NotImplementedException("CooMatrix(const MatrixType& matrix)");
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    _m   = num_rows;
    _n   = num_cols;
    _nnz = num_entries;
    row_indices.resize(_nnz);
    column_indices.resize(_nnz);
    values.resize(_nnz);
  }

  // Swap the contents of two CooMatrix objects.
  void swap(CooMatrix &matrix) {
    // TODO: CooMatrix.swap
    Morpheus::NotImplementedException(
        "CooMatrix.swap(const MatrixType& matrix)");
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  CooMatrix &operator=(const MatrixType &matrix) {
    std::cout << "CooMatrix.operator=(const MatrixType& matrix)" << std::endl;
  }

  // Operations specific to COO format

  // Sort matrix elements by row index
  void sort_by_row(void) {
    // TODO: CooMatrix.sort_by_row
    Morpheus::NotImplementedException("CooMatrix.sort_by_row()");
  }

  // Sort matrix elements by row and column index
  void sort_by_row_and_column(void) {
    // TODO: CooMatrix.sort_by_row_and_column
    Morpheus::NotImplementedException("CooMatrix.sort_by_row_and_column()");
  }

  // Determine whether matrix elements are sorted by row index
  bool is_sorted_by_row(void) {
    // TODO: CooMatrix.is_sorted_by_row
    Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row()");
    return true;
  }

  // Determine whether matrix elements are sorted by row and column index
  bool is_sorted_by_row_and_column(void) {
    // TODO: CooMatrix.is_sorted_by_row_and_column
    Morpheus::NotImplementedException(
        "CooMatrix.is_sorted_by_row_and_column()");
    return true;
  }

  // Unified routines across all formats

  inline std::string name() const { return _name; }
  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _nnz; }

 private:
  std::string _name;
  index_type _m, _n, _nnz;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_COO_MATRIX_HPP