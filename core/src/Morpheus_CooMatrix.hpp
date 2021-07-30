/**
 * Morpheus_CooMatrix.hpp
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

#ifndef MORPHEUS_COOMATRIX_HPP
#define MORPHEUS_COOMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_Sort.hpp>
#include <Morpheus_Copy.hpp>

#include <impl/Morpheus_MatrixBase.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class CooMatrix : public Impl::MatrixBase<ValueType, Properties...> {
 public:
  using type   = CooMatrix<ValueType, Properties...>;
  using traits = Impl::ContainerTraits<ValueType, Properties...>;
  using base   = Impl::MatrixBase<ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::CooTag>::tag;

  using value_type = typename traits::value_type;
  using index_type = typename traits::index_type;
  using size_type  = typename traits::index_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using HostMirror = CooMatrix<
      typename traits::non_const_value_type, typename traits::index_type,
      typename traits::array_layout,
      Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                     typename traits::host_mirror_space::memory_space>>;

  using host_mirror_type =
      CooMatrix<typename traits::non_const_value_type,
                typename traits::index_type, typename traits::array_layout,
                typename traits::host_mirror_space>;

  using pointer         = CooMatrix *;
  using const_pointer   = const CooMatrix *;
  using reference       = CooMatrix &;
  using const_reference = const CooMatrix &;

  using index_array_type      = Morpheus::vector<index_type, device_type>;
  using value_array_type      = Morpheus::vector<value_type, device_type>;
  using value_array_pointer   = typename value_array_type::pointer;
  using value_array_reference = typename value_array_type::reference;

  index_array_type row_indices, column_indices;
  value_array_type values;

  ~CooMatrix()                 = default;
  CooMatrix(const CooMatrix &) = default;
  CooMatrix(CooMatrix &&)      = default;
  reference operator=(const CooMatrix &) = default;
  reference operator=(CooMatrix &&) = default;

  // Construct an empty CooMatrix
  inline CooMatrix()
      : base("CooMatrix"), row_indices(0), column_indices(0), values(0) {}

  // Construct a CooMatrix with a specific shape and number of non-zero entries
  inline CooMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : base("CooMatrix", num_rows, num_cols, num_entries),
        row_indices(num_entries),
        column_indices(num_entries),
        values(num_entries) {}

  inline CooMatrix(const std::string name, const index_array_type &rind,
                   const index_array_type &cind, const value_array_type &vals)
      : base(name + "CooMatrix", rind.size(), cind.size(), vals.size()),
        row_indices(rind),
        column_indices(cind),
        values(vals) {}

  inline CooMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries)
      : base(name + "CooMatrix", num_rows, num_cols, num_entries),
        row_indices(num_entries),
        column_indices(num_entries),
        values(num_entries) {}

  // Construct from another matrix type
  template <typename MatrixType>
  CooMatrix(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
  }

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    row_indices.resize(num_entries);
    column_indices.resize(num_entries);
    values.resize(num_entries);
  }

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &matrix) {
    Morpheus::copy(matrix, *this);
    return *this;
  }

  // Sort matrix elements by row index
  void sort_by_row(void) {
    // TODO: CooMatrix.sort_by_row
    throw Morpheus::NotImplementedException("CooMatrix.sort_by_row()");
  }

  // Sort matrix elements by row and column index
  void sort_by_row_and_column(void) { Morpheus::sort_by_row_and_column(*this); }

  // Determine whether matrix elements are sorted by row index
  bool is_sorted_by_row(void) {
    // TODO: CooMatrix.is_sorted_by_row
    throw Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row()");
    return true;
  }

  // Determine whether matrix elements are sorted by row and column index
  bool is_sorted(void) { return Morpheus::is_sorted(*this); }
};
}  // namespace Morpheus

#endif  // MORPHEUS_COOMATRIX_HPP