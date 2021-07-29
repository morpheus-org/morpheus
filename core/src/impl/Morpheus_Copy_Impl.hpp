/**
 * Morpheus_Copy_Impl.hpp
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

#ifndef MORPHEUS_COPY_IMPL_HPP
#define MORPHEUS_COPY_IMPL_HPP

// TODO: Remove use of set during Coo to Dia Conversion
#include <set>
#include <variant>  // visit

#include <Morpheus_Core.hpp>
#include <Morpheus_Exceptions.hpp>

#include <impl/Morpheus_FormatTags.hpp>
#include <fwd/Morpheus_Fwd_CooMatrix.hpp>

namespace Morpheus {

// forward decl
template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst);

namespace Impl {

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DenseVectorTag,
          DenseVectorTag) {
  using IndexType      = typename SourceType::index_type;
  const IndexType size = src.size();
  dst.resize(size);
  // Kokkos has src and dst the other way round
  Kokkos::deep_copy(dst.view(), src.view());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DenseMatrixTag,
          DenseVectorTag) {
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows() * src.ncols());

  for (IndexType i = 0; i < src.nrows(); i++) {
    for (IndexType j = 0; j < src.ncols(); j++) {
      IndexType idx = i * src.ncols() + j;
      dst(idx)      = src(i, j);
    }
  }
}

// convert src -> coo_matrix -> dst
template <typename SourceType, typename DestinationType, typename Format1,
          typename Format2>
void copy(const SourceType& src, DestinationType& dst, Format1, Format2,
          typename std::enable_if_t<!std::is_same_v<Format1, DynamicTag> &&
                                    !std::is_same_v<Format2, DynamicTag>>* =
              nullptr) {
  using ValueType   = typename SourceType::value_type;
  using IndexType   = typename SourceType::index_type;
  using ArrayLayout = typename SourceType::array_layout;
  using MemorySpace = typename SourceType::memory_space;

  using Coo =
      Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout, MemorySpace>;
  Coo tmp;

  Morpheus::Impl::copy(src, tmp, Format1(), typename Coo::tag());
  Morpheus::Impl::copy(tmp, dst, typename Coo::tag(), Format2());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DenseMatrixTag,
          DenseMatrixTag) {
  using IndexType      = typename SourceType::index_type;
  const IndexType rows = src.nrows();
  const IndexType cols = src.ncols();
  dst.resize(rows, cols);
  // Kokkos has src and dst the other way round
  Kokkos::deep_copy(dst.view(), src.view());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DenseMatrixTag, CooTag) {
  using IndexType = typename SourceType::index_type;
  using ValueType = typename SourceType::value_type;

  // Count non-zeros
  IndexType nnz = 0;
  for (IndexType i = 0; i < src.nrows(); i++) {
    for (IndexType j = 0; j < src.ncols(); j++) {
      if (src(i, j) != ValueType(0)) nnz = nnz + 1;
    }
  }

  dst.resize(src.nrows(), src.ncols(), nnz);

  for (IndexType i = 0, n = 0; i < src.nrows(); i++) {
    for (IndexType j = 0; j < src.ncols(); j++) {
      if (src(i, j) != ValueType(0)) {
        dst.row_indices[n]    = i;
        dst.column_indices[n] = j;
        dst.values[n]         = src(i, j);
        n                     = n + 1;
      }
    }
  }
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CooTag, DenseMatrixTag) {
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols());

  for (IndexType n = 0; n < src.nnnz(); n++) {
    IndexType i = src.row_indices[n];
    IndexType j = src.column_indices[n];
    dst(i, j)   = src.values[n];
  }
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CooTag, CooTag) {
  using IndexType      = typename SourceType::index_type;
  const IndexType rows = src.nrows();
  const IndexType cols = src.ncols();
  const IndexType nnzs = src.nnnz();
  dst.resize(rows, cols, nnzs);

  Morpheus::copy(src.row_indices, dst.row_indices);
  Morpheus::copy(src.column_indices, dst.column_indices);
  Morpheus::copy(src.values, dst.values);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CsrTag, CsrTag) {
  using IndexType      = typename SourceType::index_type;
  const IndexType rows = src.nrows();
  const IndexType cols = src.ncols();
  const IndexType nnzs = src.nnnz();
  dst.resize(rows, cols, nnzs);

  Morpheus::copy(src.row_offsets, dst.row_offsets);
  Morpheus::copy(src.column_indices, dst.column_indices);
  Morpheus::copy(src.values, dst.values);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CsrTag, CooTag) {
  // Complexity: Linear.  Specifically O(nnz(csr) + max(n_row,n_col))
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // expand compressed indices
  for (IndexType i = 0; i < src.nrows(); i++) {
    for (IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i + 1]; jj++) {
      dst.row_indices[jj] = i;
    }
  }

  Morpheus::copy(src.column_indices, dst.column_indices);
  Morpheus::copy(src.values, dst.values);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CooTag, CsrTag) {
  // Complexity: Linear.  Specifically O(nnz(coo) + max(n_row,n_col))
  using IndexType = typename SourceType::index_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  // compute number of non-zero entries per row of coo src
  for (IndexType n = 0; n < src.nnnz(); n++) {
    dst.row_offsets[src.row_indices[n]]++;
  }

  // cumsum the nnz per row to get csr row_offsets
  for (IndexType i = 0, cumsum = 0; i < src.nrows(); i++) {
    IndexType temp     = dst.row_offsets[i];
    dst.row_offsets[i] = cumsum;
    cumsum += temp;
  }
  dst.row_offsets[src.nrows()] = src.nnnz();

  // write coo column indices and values into csr
  for (IndexType n = 0; n < src.nnnz(); n++) {
    IndexType row  = src.row_indices[n];
    IndexType dest = dst.row_offsets[row];

    dst.column_indices[dest] = src.column_indices[n];
    dst.values[dest]         = src.values[n];

    dst.row_offsets[row]++;
  }

  for (IndexType i = 0, last = 0; i <= src.nrows(); i++) {
    IndexType temp     = dst.row_offsets[i];
    dst.row_offsets[i] = last;
    last               = temp;
  }

  // TODO: remove duplicates, if any?
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DiaTag, DiaTag) {
  using IndexType          = typename SourceType::index_type;
  const IndexType rows     = src.nrows();
  const IndexType cols     = src.ncols();
  const IndexType nnzs     = src.nnnz();
  const IndexType off_size = src.diagonal_offsets.size();
  dst.resize(rows, cols, nnzs, off_size);

  Morpheus::copy(src.diagonal_offsets, dst.diagonal_offsets);
  Morpheus::copy(src.values, dst.values);
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DiaTag, CooTag) {
  using IndexType = typename SourceType::index_type;
  using ValueType = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  const IndexType ndiag = src.values.ncols();

  for (IndexType i = 0, nnzid = 0; i < ndiag; i++) {
    const IndexType k = src.diagonal_offsets[i];

    const IndexType i_start = std::max<IndexType>(0, -k);
    const IndexType j_start = std::max<IndexType>(0, k);

    // number of elements to process in this diagonal
    const IndexType N = std::min(src.nrows() - i_start, src.ncols() - j_start);

    for (IndexType n = 0; n < N; n++) {
      const ValueType temp = src.values(i_start + n, i);

      if (temp != ValueType(0)) {
        dst.row_indices[nnzid]    = i_start + n;
        dst.column_indices[nnzid] = j_start + n;
        dst.values[nnzid]         = temp;
        nnzid                     = nnzid + 1;
      }
    }
  }

  if (!dst.is_sorted()) {
    dst.sort_by_row_and_column();
  }
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, CooTag, DiaTag,
          typename SourceType::index_type alignment = 32) {
  using IndexType      = typename SourceType::index_type;
  using IndexArrayType = typename SourceType::index_array_type;

  if (src.nnnz() == 0) {
    dst.resize(src.nrows(), src.ncols(), src.nnnz(), 0);
    return;
  }

  IndexArrayType diag_map(src.nnnz(), 0);

  // Find on which diagonal each entry sits on
  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    diag_map[n] = src.column_indices[n] - src.row_indices[n];
  }

  // Create unique diagonal set
  std::set<IndexType> diag_set(diag_map.begin(), diag_map.end());
  IndexType ndiags = IndexType(diag_set.size());

  // TODO: Check if fill in exceeds a tolerance value otherwise throw
  // format_conversion_exception

  IndexArrayType diagonal_offsets(ndiags, 0);
  for (auto it = diag_set.cbegin(); it != diag_set.cend(); ++it) {
    auto i              = std::distance(diag_set.cbegin(), it);
    diagonal_offsets[i] = *it;
  }

  // Create diagonal indexes
  IndexArrayType diag_idx(src.nnnz(), 0);
  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    for (IndexType i = 0; i < IndexType(ndiags); i++) {
      if (diag_map[n] == diagonal_offsets[i]) diag_idx[n] = i;
    }
  }

  dst.resize(src.nrows(), src.ncols(), src.nnnz(), ndiags, alignment);

  for (IndexType i = 0; i < IndexType(ndiags); i++) {
    dst.diagonal_offsets[i] = diagonal_offsets[i];
  }

  for (IndexType n = 0; n < IndexType(src.nnnz()); n++) {
    dst.values(src.row_indices[n], diag_idx[n]) = src.values[n];
  }
}

struct copy_fn {
  using result_type = void;

  template <typename SourceType, typename DestinationType>
  result_type operator()(const SourceType& src, DestinationType& dst) {
    Morpheus::copy(src, dst);
  }
};

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DynamicTag,
          SparseMatTag) {
  auto f = std::bind(Impl::copy_fn(), std::placeholders::_1, std::ref(dst));
  std::visit(f, src.formats());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, SparseMatTag,
          DynamicTag) {
  auto f = std::bind(Impl::copy_fn(), std::cref(src), std::placeholders::_1);
  std::visit(f, dst.formats());
}

template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst, DynamicTag, DynamicTag) {
  std::visit(Impl::copy_fn(), src.formats(), dst.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_COPY_IMPL_HPP