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

#include <impl/Morpheus_MatrixBase.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
class CsrMatrix : public Impl::MatrixBase<CsrMatrix, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<CsrMatrix, ValueType, Properties...>;
  using type   = typename traits::type;
  using base   = Impl::MatrixBase<CsrMatrix, ValueType, Properties...>;
  using tag    = typename MatrixFormatTag<Morpheus::CsrTag>::tag;

  using value_type           = typename traits::value_type;
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::index_type;
  using index_type           = typename traits::index_type;
  using non_const_index_type = typename traits::non_const_index_type;

  using memory_space = typename traits::memory_space;
  using HostMirror   = typename traits::HostMirror;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using index_array_type =
      Morpheus::DenseVector<index_type, index_type, memory_space>;
  using index_array_pointer = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;

  using value_array_type =
      Morpheus::DenseVector<value_type, index_type, memory_space>;
  using value_array_pointer = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;

  index_array_type row_offsets, column_indices;
  value_array_type values;

  ~CsrMatrix()                 = default;
  CsrMatrix(const CsrMatrix &) = default;
  CsrMatrix(CsrMatrix &&)      = default;
  CsrMatrix &operator=(const CsrMatrix &) = default;
  CsrMatrix &operator=(CsrMatrix &&) = default;

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
                   const index_type num_cols, const index_type num_entries,
                   const index_array_type &roff, const index_array_type &cind,
                   const value_array_type &vals)
      : base(name + "CsrMatrix", num_rows, num_cols, num_entries),
        row_offsets(roff),
        column_indices(cind),
        values(vals) {}

  inline CsrMatrix(const std::string name, const index_type num_rows,
                   const index_type num_cols, const index_type num_entries)
      : base(name + "CsrMatrix", num_rows, num_cols, num_entries),
        row_offsets(num_rows + 1),
        column_indices(num_entries),
        values(num_entries) {}

  // Construct from another matrix type (Shallow)
  // Needs to be a compatible type
  template <class VR, class... PR>
  CsrMatrix(const CsrMatrix<VR, PR...> &src,
            typename std::enable_if<is_compatible_type<
                CsrMatrix, CsrMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.name() + "(ShallowCopy)", src.nrows(), src.ncols(),
             src.nnnz()),
        row_offsets(src.row_offsets),
        column_indices(src.column_indices),
        values(src.values) {}

  // Assignment from another matrix type (Shallow)
  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<CsrMatrix, CsrMatrix<VR, PR...>>::value,
      CsrMatrix &>::type
  operator=(const CsrMatrix<VR, PR...> &src) {
    this->set_name(src.name());
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    row_offsets    = src.row_offsets;
    column_indices = src.column_indices;
    values         = src.values;

    return *this;
  }

  // Construct from another matrix type
  template <typename MatrixType>
  CsrMatrix(const MatrixType &src) = delete;

  // Assignment from another matrix type
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  // Resize matrix dimensions and underlying storage
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    row_offsets.resize(num_rows + 1);
    column_indices.resize(num_entries);
    values.resize(num_entries);
  }

  template <class VR, class... PR>
  inline CsrMatrix &allocate(const std::string name,
                             const CsrMatrix<VR, PR...> &src) {
    this->set_name(name);
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }
};
}  // namespace Morpheus

#endif  // MORPHEUS_CSRMATRIX_HPP