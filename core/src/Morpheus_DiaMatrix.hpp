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
#include <Morpheus_Copy.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

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
  using size_type            = typename traits::index_type;
  using index_type           = typename traits::index_type;
  using non_const_index_type = typename traits::non_const_index_type;
  using array_layout         = typename traits::array_layout;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using HostMirror       = typename traits::HostMirror;
  using host_mirror_type = typename traits::host_mirror_type;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using index_array_type = Morpheus::vector<index_type, device_type>;
  using value_array_type =
      Morpheus::DenseMatrix<value_type, index_type, Kokkos::LayoutLeft,
                            device_type>;
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
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

  inline DiaMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries,
                   const index_type num_diagonals,
                   const index_type alignment = 32)
      : base(name + "DiaMatrix", num_rows, num_cols, num_entries),
        diagonal_offsets(num_diagonals) {
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

  // !FIXME: Remove deep copy and perform shallow copy
  // Construct from another matrix type
  template <class VR, class... PR>
  DiaMatrix(const DiaMatrix<VR, PR...> &matrix) {
    Morpheus::copy(matrix, *this);
  }

  // !FIXME: Remove deep copy and perform shallow copy
  // Assignment from another matrix type
  template <class VR, class... PR>
  reference operator=(const DiaMatrix<VR, PR...> &matrix) {
    Morpheus::copy(matrix, *this);
    return *this;
  }

  // !FIXME: Needs to perform conversion
  // Construct from another matrix type
  template <typename MatrixType>
  DiaMatrix(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
  }

  // !FIXME: Needs to perform conversion
  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
    return *this;
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals,
                     const index_type alignment = 32) {
    base::resize(num_rows, num_cols, num_entries);
    diagonal_offsets.resize(num_diagonals);
    values.resize(this->_pad_size(num_rows, alignment), num_diagonals);
  }

 private:
  // Calculates padding to align the data based on the current diagonal length
  inline const index_type _pad_size(index_type diag_len, index_type alignment) {
    return alignment * ((diag_len + alignment - 1) / alignment);
  }
};
}  // namespace Morpheus

#endif  // MORPHEUS_DIAMATRIX_HPP