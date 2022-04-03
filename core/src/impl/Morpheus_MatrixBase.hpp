/**
 * Morpheus_MatrixBase.hpp
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
#ifndef MORPHEUS_MATRIXBASE_HPP
#define MORPHEUS_MATRIXBASE_HPP

#include <Morpheus_TypeTraits.hpp>
#include <impl/Morpheus_ContainerTraits.hpp>

namespace Morpheus {

namespace Impl {

template <template <class, class...> class Container, class ValueType,
          class... Properties>
class MatrixBase : public ContainerTraits<Container, ValueType, Properties...> {
 public:
  using type       = MatrixBase<Container, ValueType, Properties...>;
  using traits     = ContainerTraits<Container, ValueType, Properties...>;
  using index_type = typename traits::index_type;

  MatrixBase() : _m(0), _n(0), _nnz(0) {}

  template <typename Matrix>
  MatrixBase(const Matrix& m,
             typename std::enable_if<is_matrix_v<Matrix>>::type* = 0)
      : _m(m.nrows()), _n(m.ncols()), _nnz(m.nnnz()) {}

  MatrixBase(index_type rows, index_type cols, index_type entries = 0)
      : _m(rows), _n(cols), _nnz(entries) {}

  void resize(index_type rows, index_type cols, index_type entries) {
    _m   = rows;
    _n   = cols;
    _nnz = entries;
  }

  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _nnz; }
  inline void set_nrows(const index_type rows) { _m = rows; }
  inline void set_ncols(const index_type cols) { _n = cols; }
  inline void set_nnnz(const index_type nnz) { _nnz = nnz; }

 private:
  index_type _m, _n, _nnz;

};  // namespace Impl
}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXBASE_HPP