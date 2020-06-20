/*****************************************************************************
 *
 *  ell_matrix.inl
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

/*! \file ell_matrix.inl
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_DETAIL_ELL_MATRIX_INL
#define MORPHEUS_MATRIX_FORMATS_DETAIL_ELL_MATRIX_INL

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail

	template<typename IndexType, typename ValueType, class MemorySpace>
	ell_matrix<IndexType, ValueType, MemorySpace>
	::ell_matrix(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
	           const size_t num_entries_per_row, const size_t alignment)
	           : parent_t(num_rows, num_cols, num_nnz, num_entries_per_row, alignment)
	{}

	template<typename IndexType, typename ValueType, class MemorySpace>
	template<typename MatrixType>
	ell_matrix<IndexType, ValueType, MemorySpace>
	::ell_matrix(const MatrixType& matrix) : parent_t(matrix)
	{}

	template<typename IndexType, typename ValueType, class MemorySpace>
	template<typename MatrixType>
	typename ell_matrix<IndexType, ValueType, MemorySpace>::reference
	ell_matrix<IndexType, ValueType, MemorySpace>
	::operator = (const MatrixType& matrix)
	{
		parent_t::operator=(matrix);
		return *this;
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	void
	ell_matrix<IndexType, ValueType, MemorySpace>
	::resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
	            const size_t num_entries_per_row)
	{
		parent_t::resize(num_rows, num_cols, num_nnz, num_entries_per_row);
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	void
	ell_matrix<IndexType, ValueType, MemorySpace>
	::resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
	            const size_t num_entries_per_row, const size_t alignment)
	{
		parent_t::resize(num_rows, num_cols, num_nnz, num_entries_per_row, alignment);
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	void
	ell_matrix<IndexType, ValueType, MemorySpace>
	::swap(ell_matrix& matrix)
	{
		parent_t::swap(matrix);
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	size_t
	ell_matrix<IndexType, ValueType, MemorySpace>::nrows()
	{
		return parent_t::num_rows;
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	size_t
	ell_matrix<IndexType, ValueType, MemorySpace>::ncols()
	{
		return parent_t::num_cols;
	}

	template<typename IndexType, typename ValueType, class MemorySpace>
	size_t
	ell_matrix<IndexType, ValueType, MemorySpace>::nnz()
	{
		return parent_t::num_entries;
	}

}   // end namespace morpheus

#endif //MORPHEUS_MATRIX_FORMATS_DETAIL_ELL_MATRIX_INL
