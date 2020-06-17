/*****************************************************************************
 *
 *  dense_matrix.inl
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

/*! \file dense_matrix.inl
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_MATRIX_INL
#define MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_MATRIX_INL

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail

	template<typename ValueType, typename MemorySpace, typename Orientation>
	void
	dense_matrix<ValueType, MemorySpace, Orientation>
    ::resize(const size_t num_rows, const size_t num_cols)
	{
		parent_t::resize(num_rows, num_cols);
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	void
	dense_matrix<ValueType, MemorySpace, Orientation>
    ::swap(dense_matrix& matrix)
	{
		parent_t::swap(matrix);
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	typename dense_matrix<ValueType,MemorySpace,Orientation>::reference
	dense_matrix<ValueType,MemorySpace,Orientation>
    ::operator=(const dense_matrix& matrix)
	{
		parent_t::operator=(matrix);
		return *this;
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	template <typename MatrixType>
	typename dense_matrix<ValueType,MemorySpace,Orientation>::reference
	dense_matrix<ValueType,MemorySpace,Orientation>
	::operator=(const MatrixType& matrix)
	{
		parent_t::operator=(matrix);
		return *this;
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	size_t
	dense_matrix<ValueType,MemorySpace,Orientation>::nrows()
	{
		return parent_t::num_rows;
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	size_t
	dense_matrix<ValueType,MemorySpace,Orientation>::ncols()
	{
		return parent_t::num_cols;
	}

	template<typename ValueType, typename MemorySpace, typename Orientation>
	size_t
	dense_matrix<ValueType,MemorySpace,Orientation>::nnz()
	{
		return parent_t::num_entries;
	}

}   // end namespace morpheus

#endif //MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_MATRIX_INL
