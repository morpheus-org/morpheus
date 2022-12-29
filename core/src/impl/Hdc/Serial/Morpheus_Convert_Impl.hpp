/**
 * Morpheus_Convert_Impl.hpp
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

#ifndef MORPHEUS_HDC_SERIAL_CONVERT_IMPL_HPP
#define MORPHEUS_HDC_SERIAL_CONVERT_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_SpaceTraits.hpp>
#include <Morpheus_FormatTraits.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Spaces.hpp>

#include <impl/Dia/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Csr/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Serial/Morpheus_Convert_Impl.hpp>
#include <impl/Coo/Serial/Morpheus_Sort_Impl.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_hdc_matrix_format_container_v<SourceType> &&
        Morpheus::is_hdc_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  dst.set_nrows(src.nrows());
  dst.set_ncols(src.ncols());
  dst.set_nnnz(src.nnnz());
  dst.set_alignment(src.alignment());

  Impl::convert<ExecSpace>(src.cdia(), dst.dia());
  Impl::convert<ExecSpace>(src.ccsr(), dst.csr());
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_hdc_matrix_format_container_v<SourceType> &&
        Morpheus::is_coo_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;
  using value_type = typename SourceType::value_type;

  dst.resize(src.nrows(), src.ncols(), src.nnnz());

  const size_type ndiag = src.cdia().cvalues().ncols();

  size_type nnzid = 0;
  for (size_type i = 0; i < ndiag; i++) {
    const index_type k = src.cdia().cdiagonal_offsets(i);

    const size_type i_start = std::max<index_type>(0, -k);
    const size_type j_start = std::max<index_type>(0, k);

    // number of elements to process in this diagonal
    const size_type N = std::min(src.nrows() - i_start, src.ncols() - j_start);

    for (size_type n = 0; n < N; n++) {
      const value_type temp = src.cdia().cvalues(i_start + n, i);

      if (temp != value_type(0)) {
        dst.row_indices(nnzid)    = i_start + n;
        dst.column_indices(nnzid) = j_start + n;
        dst.values(nnzid)         = temp;
        nnzid++;
      }
    }
  }

  // expand compressed indices
  for (size_type i = 0; i < src.ccsr().nrows(); i++) {
    for (index_type jj = src.ccsr().crow_offsets(i);
         jj < src.ccsr().crow_offsets(i + 1); jj++) {
      dst.row_indices(jj + nnzid) = i;
    }
  }

  for (size_type n = 0; n < src.ccsr().nnnz(); n++) {
    dst.column_indices(n + nnzid) = src.ccsr().ccolumn_indices(n);
  }

  for (size_type n = 0; n < src.ccsr().nnnz(); n++) {
    dst.values(n + nnzid) = src.ccsr().cvalues(n);
  }

  if (!Impl::is_sorted<ExecSpace>(dst)) {
    Impl::sort_by_row_and_column<ExecSpace>(dst);
  }
}

template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(
    const SourceType& src, DestinationType& dst,
    typename std::enable_if<
        Morpheus::is_coo_matrix_format_container_v<SourceType> &&
        Morpheus::is_hdc_matrix_format_container_v<DestinationType> &&
        Morpheus::has_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, SourceType, DestinationType>>::type* =
        nullptr) {
  using size_type  = typename SourceType::size_type;
  using index_type = typename SourceType::index_type;

  size_type min_diag_elem = src.nrows() / 2;
  std::vector<index_type> diag_map(src.nnnz(), 0);

  // Find on which diagonal each entry sits on
  for (size_type n = 0; n < src.nnnz(); n++) {
    diag_map[n] = src.ccolumn_indices(n) - src.crow_indices(n);
  }

  size_type dia_nnz = 0, csr_nnz = 0;
  // Create unique diagonal set
  std::set<index_type> diag_set(diag_map.begin(), diag_map.end());

  // Count non-zeros for each part and erase any diags from set that go into CSR
  for (size_type key = 0; key <= diag_set.size(); key++) {
    index_type diag_elements =
        std::count(diag_map.begin(), diag_map.end(), diag_set(key));
    if (diag_elements < min_diag_elem) {
      csr_nnz += diag_elements;
      diag_set.erase(key);
    } else {
      dia_nnz += diag_elements;
    }
  }

  size_type ndiags          = diag_set.size();
  const size_type alignment = 32;

  dst.resize(src.nrows(), src.ncols(), dia_nnz, csr_nnz, ndiags, alignment);
  dst.csr().row_offsets().assign(src.nrows() + 1, 0);
  for (auto it = diag_set.cbegin(); it != diag_set.cend(); ++it) {
    auto i                        = std::distance(diag_set.cbegin(), it);
    dst.dia().diagonal_offsets(i) = *it;
  }

  // Write the DIA and CSR parts
  for (size_type n = 0; n < src.nnnz(); n++) {
    for (size_type i = 0; i < ndiags; i++) {
      if (diag_map[n] == dst.dia().diagonal_offsets(i)) {
        dst.dia().values(src.crow_indices(n), i) = src.cvalues(n);
        break;
      } else {
        dst.csr().row_offsets(src.crow_indices(n))++;
        dst.csr().column_indices(n) = src.ccolumn_indices(n);
        dst.csr().values(n)         = src.cvalues(n);
      }
    }
  }

  // Compress the CSR part row_offsets
  index_type cumsum = 0;
  for (size_type i = 0; i < src.nrows(); i++) {
    index_type temp          = dst.csr().row_offsets(i);
    dst.csr().row_offsets(i) = cumsum;
    cumsum += temp;
  }

  dst.csr().row_offsets(src.nrows()) = csr_nnz;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_HDC_SERIAL_CONVERT_IMPL_HPP