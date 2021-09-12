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
  using HostMirror      = typename traits::HostMirror;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using index_array_type =
      Morpheus::DenseVector<index_type, index_type, array_layout, memory_space>;
  using index_array_pointer = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;

  using value_array_type =
      Morpheus::DenseMatrix<value_type, index_type, array_layout, memory_space>;
  using value_array_pointer = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;

  index_type ndiags, nalign;
  index_array_type diagonal_offsets;
  value_array_type values;

  ~DiaMatrix()                 = default;
  DiaMatrix(const DiaMatrix &) = default;
  DiaMatrix(DiaMatrix &&)      = default;
  DiaMatrix &operator=(const DiaMatrix &) = default;
  DiaMatrix &operator=(DiaMatrix &&) = default;

  // Construct an empty DiaMatrix
  inline DiaMatrix() : base("DiaMatrix"), diagonal_offsets(), values() {}

  // Construct a DiaMatrix with:
  //      a specific shape
  //      number of non-zero entries
  //      number of occupied diagonals
  //      amount of padding used to align the data (default=32)
  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries, const index_type num_diagonals,
                   const index_type alignment = 32)
      : base("DiaMatrix", num_rows, num_cols, num_entries),
        diagonal_offsets(num_diagonals) {
    ndiags = num_diagonals;
    nalign = alignment;
    values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                  num_diagonals);
  }

  inline DiaMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries,
                   const index_array_type &diag_offsets,
                   const value_array_type &vals)
      : base(name + "DiaMatrix", num_rows, num_cols, num_entries),
        diagonal_offsets(diag_offsets),
        values(vals) {
    ndiags = diagonal_offsets.size();
    nalign = values.nrows();
  }

  inline DiaMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries,
                   const index_type num_diagonals,
                   const index_type alignment = 32)
      : base(name + "DiaMatrix", num_rows, num_cols, num_entries),
        diagonal_offsets(num_diagonals) {
    ndiags = num_diagonals;
    nalign = alignment;
    values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                  num_diagonals);
  }

  // Construct from another matrix type (Shallow)
  template <class VR, class... PR>
  DiaMatrix(const DiaMatrix<VR, PR...> &src,
            typename std::enable_if<is_compatible_type<
                DiaMatrix, DiaMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.name() + "(ShallowCopy)", src.nrows(), src.ncols(),
             src.nnnz()),
        ndiags(src.ndiags),
        nalign(src.nalign),
        diagonal_offsets(src.diagonal_offsets),
        values(src.values) {}

  // Assignment from another matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<DiaMatrix, DiaMatrix<VR, PR...>>::value,
      DiaMatrix &>::type
  operator=(const DiaMatrix<VR, PR...> &src) {
    this->set_name(src.name());
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    ndiags           = src.ndiags;
    nalign           = src.nalign;
    diagonal_offsets = src.diagonal_offsets;
    values           = src.values;

    return *this;
  }

  // Construct from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  DiaMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_compatible_container<
                DiaMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.name() + "(ShallowCopy)", src.nrows(), src.ncols(),
             src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  // Assignment from a compatible dynamic matrix type (Shallow)
  // Throws when active type of dynamic matrix not same to concrete type
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_container<DiaMatrix, DynamicMatrix<VR, PR...>>::value,
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
    ndiags = num_diagonals;
    nalign = alignment;
    diagonal_offsets.resize(num_diagonals);
    values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                  num_diagonals);
  }

  template <class VR, class... PR>
  inline DiaMatrix &allocate(const std::string name,
                             const DiaMatrix<VR, PR...> &src) {
    this->set_name(name);
    resize(src.nrows(), src.ncols(), src.nnnz(), src.ndiags, src.nalign);
    return *this;
  }

  formats_e format_enum() const { return Morpheus::DIA_FORMAT; }

  int format_index() const { return static_cast<int>(Morpheus::DIA_FORMAT); }
};

}  // namespace Morpheus

#endif  // MORPHEUS_DIAMATRIX_HPP