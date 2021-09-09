/**
 * Morpheus_DenseMatrix.hpp
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

#ifndef MORPHEUS_DENSEMATRIX_HPP
#define MORPHEUS_DENSEMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

#include <Kokkos_Core.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class DenseMatrix
    : public Impl::MatrixBase<DenseMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<DenseMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<DenseMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::DenseMatrixTag>::tag;

  using value_type           = typename traits::value_type;
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::index_type;
  using index_type           = typename traits::index_type;
  using non_const_index_type = typename traits::non_const_index_type;

  using array_layout    = typename traits::array_layout;
  using memory_space    = typename traits::memory_space;
  using execution_space = typename memory_space::execution_space;
  using HostMirror      = typename traits::HostMirror;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using value_array_type =
      Kokkos::View<value_type **, array_layout, memory_space>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

  ~DenseMatrix()                   = default;
  DenseMatrix(const DenseMatrix &) = default;
  DenseMatrix(DenseMatrix &&)      = default;
  DenseMatrix &operator=(const DenseMatrix &) = default;
  DenseMatrix &operator=(DenseMatrix &&) = default;

  // Construct an empty DenseMatrix
  inline DenseMatrix() : base("DenseMatrix"), _values() {}

  // Construct a DenseMatrix with a specific shape
  inline DenseMatrix(const index_type num_rows, const index_type num_cols,
                     const value_type val = 0)
      : base("DenseMatrix", num_rows, num_cols, num_rows * num_cols),
        _values("DenseMatrix", size_t(num_rows), size_t(num_cols)) {
    assign(num_rows, num_cols, val);
  }

  inline DenseMatrix(const std::string name, const index_type num_rows,
                     const index_type num_cols, const value_type val = 0)
      : base(name + "DenseMatrix", num_rows, num_cols, num_rows * num_cols),
        _values(name + "(DenseMatrix)", size_t(num_rows), size_t(num_cols)) {
    assign(num_rows, num_cols, val);
  }

  // Construct from another dense matrix type (Shallow)
  template <class VR, class... PR>
  inline DenseMatrix(
      const DenseMatrix<VR, PR...> &src,
      typename std::enable_if<is_compatible_type<
          DenseMatrix, DenseMatrix<VR, PR...>>::value>::type * = nullptr)
      : base("ShallowDenseMatrix" + src.name(), src.nrows(), src.ncols(),
             src.nrows() * src.ncols()),
        _values(src.view()) {}

  // Assignment from another dense matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<DenseMatrix, DenseMatrix<VR, PR...>>::value,
      DenseMatrix &>::type
  operator=(const DenseMatrix<VR, PR...> &src) {
    if (this != &src) {
      set_name(src.name());
      set_nrows(src.nrows());
      set_ncols(src.ncols());
      set_nnnz(src.nnnz());
      _values = src.view();
    }
    return *this;
  }

  // !FIXME: Needs to perform conversion
  // Construct from another matrix type
  template <class MatrixType>
  DenseMatrix(const MatrixType &src) = delete;

  // !FIXME: Needs to perform conversion
  // Assignment from another matrix type
  template <class MatrixType>
  reference operator=(const MatrixType &src) = delete;

  inline void assign(index_type num_rows, index_type num_cols,
                     const value_type val) {
    using range_policy = Kokkos::RangePolicy<index_type, execution_space>;
    /* Resize if necessary */
    this->resize(num_rows, num_cols);

    range_policy policy(0, num_rows);
    set_functor f(_values, val, num_cols);
    Kokkos::parallel_for("Morpheus::DenseMatrix::assign", policy, f);
  }

  // Modifiers
  inline void resize(index_type num_rows, index_type num_cols) {
    base::resize(num_rows, num_cols, num_rows * num_cols);
    Kokkos::resize(_values, size_t(num_rows), size_t(num_cols));
  }

  template <class VR, class... PR>
  inline DenseMatrix &allocate(const std::string name,
                               const DenseMatrix<VR, PR...> &src) {
    this->set_name(name);
    resize(src.nrows(), src.ncols());
    return *this;
  }

  // Element access
  inline value_array_reference operator()(index_type i, index_type j) const {
    return _values(i, j);
  }

  inline value_array_pointer data() const { return _values.data(); }
  inline value_array_type &view() { return _values; }
  inline const value_array_type &const_view() const { return _values; }

 private:
  value_array_type _values;

 public:
  struct set_functor {
    value_array_type _data;
    value_type _val;
    index_type _ncols;

    set_functor(value_array_type data, value_type val, index_type ncols)
        : _data(data), _val(val), _ncols(ncols) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const index_type &i) const {
      for (index_type j = 0; j < _ncols; j++) {
        _data(i, j) = _val;
      }
    }
  };
};
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEMATRIX_HPP