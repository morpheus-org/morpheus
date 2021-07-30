/**
 * Morpheus_CsrMatrix.hpp
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

#ifndef MORPHEUS_CSRMATRIX_HPP
#define MORPHEUS_CSRMATRIX_HPP

#include <string>

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_Copy.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class CsrMatrix : public Impl::MatrixBase<ValueType, Properties...> {
 public:
  using type   = CsrMatrix<ValueType, Properties...>;
  using traits = Impl::ContainerTraits<ValueType, Properties...>;
  using base   = Impl::MatrixBase<ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::CsrTag>::tag;

  using value_type = typename traits::value_type;
  using index_type = typename traits::index_type;
  using size_type  = typename traits::index_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using HostMirror = CsrMatrix<
      typename traits::non_const_value_type, typename traits::index_type,
      typename traits::array_layout,
      Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                     typename traits::host_mirror_space::memory_space>>;

  using host_mirror_type =
      CsrMatrix<typename traits::non_const_value_type,
                typename traits::index_type, typename traits::array_layout,
                typename traits::host_mirror_space>;

  using pointer         = CsrMatrix *;
  using const_pointer   = const CsrMatrix *;
  using reference       = CsrMatrix &;
  using const_reference = const CsrMatrix &;

  using index_array_type      = Morpheus::vector<index_type, device_type>;
  using value_array_type      = Morpheus::vector<value_type, device_type>;
  using value_array_pointer   = typename value_array_type::pointer;
  using value_array_reference = typename value_array_type::reference;

  index_array_type row_offsets, column_indices;
  value_array_type values;

  ~CsrMatrix()                 = default;
  CsrMatrix(const CsrMatrix &) = default;
  CsrMatrix(CsrMatrix &&)      = default;
  reference operator=(const CsrMatrix &) = default;
  reference operator=(CsrMatrix &&) = default;

  // Construct an empty CsrMatrix
  inline CsrMatrix()
      : base("CsrMatrix"), row_offsets(1), column_indices(0), values(0) {}

  // Construct a CsrMatrix with a specific shape and number of non-zero entries
  inline CsrMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : base("CsrMatrix", num_rows, num_cols, num_entries),
        row_offsets(num_rows + 1),
        column_indices(num_entries),
        values(num_entries) {}

  inline CsrMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries)
      : base(name + "CsrMatrix", num_rows, num_cols, num_entries),
        row_offsets(num_rows + 1),
        column_indices(num_entries),
        values(num_entries) {}

  // Construct from another matrix type
  template <typename MatrixType>
  CsrMatrix(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    row_offsets.resize(num_rows + 1);
    column_indices.resize(num_entries);
    values.resize(num_entries);
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
    return *this;
  }
};
}  // namespace Morpheus

#endif  // MORPHEUS_CSRMATRIX_HPP