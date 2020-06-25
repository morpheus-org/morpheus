/*****************************************************************************
 *
 *  hyb_matrix.inl
 *
 *  Edinburgh Parallel Computing Centre (EPCC)
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Christodoulos Stylianou (s1887443@ed.ac.uk)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *****************************************************************************/

/*! \file hyb_matrix.inl
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_DETAIL_HYB_MATRIX_INL
#define MORPHEUS_MATRIX_FORMATS_DETAIL_HYB_MATRIX_INL

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail

	template <typename IndexType, typename ValueType, class MemorySpace>
	hyb_matrix<IndexType, ValueType, MemorySpace>
	::hyb_matrix(const size_t num_rows, const size_t num_cols,
	           const size_t num_ell_entries, const size_t num_coo_entries,
	           const size_t num_entries_per_row, const size_t alignment)
	           : parent_t(num_rows, num_cols, num_ell_entries, num_coo_entries,
	           		      num_entries_per_row, alignment)
	{}

	template <typename IndexType, typename ValueType, class MemorySpace>
	template<typename MatrixType>
	hyb_matrix<IndexType, ValueType, MemorySpace>
	::hyb_matrix(const MatrixType& matrix)
			: parent_t(matrix)
	{}

	template <typename IndexType, typename ValueType, class MemorySpace>
	template<typename MatrixType>
	typename hyb_matrix<IndexType, ValueType, MemorySpace>::reference
	hyb_matrix<IndexType, ValueType, MemorySpace>
	::operator = (const MatrixType& matrix)
	{
		parent_t::operator=(matrix);
		return *this;
	}

	template <typename IndexType, typename ValueType, class MemorySpace>
	void
	hyb_matrix<IndexType, ValueType, MemorySpace>
	::resize(size_t num_rows, size_t num_cols,
             size_t num_ell_entries, size_t num_coo_entries,
             size_t num_entries_per_row, size_t alignment)
	{
		parent_t::resize(num_rows, num_cols,
						 num_ell_entries, num_coo_entries,
						 num_entries_per_row, alignment);
	}

	template <typename IndexType, typename ValueType, class MemorySpace>
	void
	hyb_matrix<IndexType, ValueType, MemorySpace>
	::swap(hyb_matrix& matrix)
	{
		parent_t::swap(matrix);
	}

	template <typename IndexType, typename ValueType, class MemorySpace>
	size_t
	hyb_matrix<IndexType, ValueType, MemorySpace>::nrows()
	{
		return parent_t::ell.num_rows;
	}

	template <typename IndexType, typename ValueType, class MemorySpace>
	size_t
	hyb_matrix<IndexType, ValueType, MemorySpace>::ncols()
	{
		return parent_t::ell.num_cols;
	}

	template <typename IndexType, typename ValueType, class MemorySpace>
	size_t
	hyb_matrix<IndexType, ValueType, MemorySpace>::nnz()
	{
		return parent_t::ell.num_entries + parent_t::coo.num_entries;
	}

}   // end namespace morpheus

#endif //MORPHEUS_MATRIX_FORMATS_DETAIL_HYB_MATRIX_INL
