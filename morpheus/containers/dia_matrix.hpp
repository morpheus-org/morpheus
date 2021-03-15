/**
 * dia_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_DIA_MATRIX_HPP
#define MORPHEUS_CONTAINERS_DIA_MATRIX_HPP

#include <iostream>
#include <string>
#include <vector>

#include <morpheus/core/exceptions.hpp>
#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/matrix_tags.hpp>

namespace Morpheus {

struct DiaTag : public Impl::MatrixTag {};

template <class... Properties>
class DiaMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type       = DiaMatrix<Properties...>;
  using traits     = Impl::MatrixTraits<Properties...>;
  using index_type = typename traits::index_type;
  using value_type = typename traits::value_type;
  using tag        = typename FormatTag<DiaTag>::tag;

  // TODO: Use Morpheus::array instead of std::vector
  using index_array_type = std::vector<index_type>;
  // TODO: Use Morpheus::dense_matrix instead of std::vector
  using value_array_type = std::vector<value_type>;

  index_array_type diagonal_offsets;
  value_array_type values;

  // Construct an empty DiaMatrix
  inline DiaMatrix() {}

  // Construct a DiaMatrix with:
  //      a specific shape
  //      number of non-zero entries
  //      number of occupied diagonals
  //      amount of padding used to align the data structures (default=32)
  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries, const index_type num_diagonals,
                   const index_type alignment = 32)
      : _m(num_rows),
        _n(num_cols),
        _nnz(num_entries),
        diagonal_offsets(num_diagonals) {
    // TODO: DiaMatrix(...)
    Morpheus::NotImplementedException("DiaMatrix(...)");
  }

  // Construct from another matrix type
  template <typename MatrixType>
  DiaMatrix(const MatrixType &matrix) {
    // TODO: DiaMatrix(const MatrixType& matrix)
    Morpheus::NotImplementedException("DiaMatrix(const MatrixType& matrix)");
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals) {
    // TODO: resize(...)
    Morpheus::NotImplementedException("DiaMatrix.resize(...)");
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals,
                     const index_type alignment) {
    // TODO: resize(...)
    Morpheus::NotImplementedException("DiaMatrix.resize(...)");
  }

  // Swap the contents of two DiaMatrix objects.
  void swap(DiaMatrix &matrix) {
    // TODO: swap(DiaMatrix& matrix)
    Morpheus::NotImplementedException(
        "DiaMatrix.swap(const DiaMatrix& matrix)");
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  DiaMatrix &operator=(const MatrixType &matrix) {
    // TODO: DiaMatrix.operator=(const MatrixType& matrix)
    Morpheus::NotImplementedException(
        "DiaMatrix.operator=(const MatrixType& matrix)");
  }

  inline std::string name() const { return _name; }
  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _nnz; }

 private:
  std::string _name = "DiaMatrix";
  index_type _m, _n, _nnz;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DIA_MATRIX_HPP