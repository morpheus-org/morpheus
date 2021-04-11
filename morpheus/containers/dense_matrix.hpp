/**
 * dense_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_DENSE_MATRIX_HPP
#define MORPHEUS_CONTAINERS_DENSE_MATRIX_HPP

#include <string>

#include <morpheus/core/core.hpp>
#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/matrix_tags.hpp>
#include <morpheus/core/exceptions.hpp>

namespace Morpheus {

struct DenseMatrixTag : public Impl::DenseMatTag {};

template <class... Properties>
class DenseMatrix : public Impl::MatrixTraits<Properties...> {
 public:
  using type   = DenseMatrix<Properties...>;
  using traits = Impl::MatrixTraits<Properties...>;
  using tag    = typename MatrixFormatTag<DenseMatrixTag>::tag;

  using value_type   = typename traits::value_type;
  using index_type   = typename traits::index_type;
  using size_type    = typename traits::index_type;
  using array_layout = typename traits::array_layout;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using pointer         = DenseMatrix *;
  using const_pointer   = const DenseMatrix *;
  using reference       = DenseMatrix &;
  using const_reference = const DenseMatrix &;

  using value_array_type =
      Kokkos::View<value_type **, array_layout, memory_space>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

  ~DenseMatrix()                   = default;
  DenseMatrix(const DenseMatrix &) = default;
  DenseMatrix(DenseMatrix &&)      = default;
  reference operator=(const DenseMatrix &) = default;
  reference operator=(DenseMatrix &&) = default;

  // Construct an empty DenseMatrix
  inline DenseMatrix() : _name("DenseMatrix"), _m(0), _n(0), _values() {}

  // Construct a DenseMatrix with a specific shape
  inline DenseMatrix(const index_type num_rows, const index_type num_cols,
                     const value_type val = 0)
      : _name("DenseMatrix"),
        _m(num_rows),
        _n(num_cols),
        _values("DenseMatrix", size_t(num_rows), size_t(num_cols)) {
    assign(num_rows, num_cols, val);
  }

  inline DenseMatrix(const std::string name, const index_type num_rows,
                     const index_type num_cols, const value_type val = 0)
      : _name(name + "(DenseMatrix)"),
        _m(num_rows),
        _n(num_cols),
        _values(name + "(DenseMatrix)", size_t(num_rows), size_t(num_cols)) {
    assign(num_rows, num_cols, val);
  }

  inline void assign(index_type num_rows, index_type num_cols,
                     const value_type val) {
    /* Resize if necessary */
    using I = index_type;
    this->resize(num_rows, num_cols);

    Kokkos::RangePolicy<execution_space, size_type> range(0, num_rows);
    Kokkos::parallel_for(
        "Morpheus::DenseMatrix::assign", range, KOKKOS_LAMBDA(const I i) {
          for (I j = 0; j < num_cols; j++) {
            _values(i, j) = val;
          }
        });
  }

  // Modifiers
  inline void resize(index_type num_rows, index_type num_cols) {
    Kokkos::resize(_values, size_t(num_rows), size_t(num_cols));
    _m = num_rows;
    _n = num_cols;
  }

  // Element access
  inline value_array_reference operator()(index_type i, index_type j) const {
    return _values(i, j);
  }

  inline value_array_pointer data() const { return _values.data(); }
  inline const value_array_type &view() const { return _values; }

  // Unified routines across all formats

  inline std::string name() const { return _name; }
  inline index_type nrows() const { return _m; }
  inline index_type ncols() const { return _n; }
  inline index_type nnnz() const { return _m * _n; }
  inline void set_nrows(const index_type rows) { _m = rows; }
  inline void set_ncols(const index_type cols) { _n = cols; }

 private:
  std::string _name;
  index_type _m, _n;
  value_array_type _values;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DENSE_MATRIX_HPP