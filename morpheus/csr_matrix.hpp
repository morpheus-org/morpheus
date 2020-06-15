/*****************************************************************************
 *
 *  csr_matrix.hpp
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

/*! \file csr_matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_CSR_MATRIX_HPP
#define MORPHEUS_CSR_MATRIX_HPP

#include <cusp/csr_matrix.h>

namespace morpheus
{
	// Currently using the Cusp Interface
	template <typename IndexType, typename ValueType, class MemorySpace>
	class csr_matrix : public cusp::csr_matrix<IndexType,ValueType,MemorySpace>
	{
	private:
		using parent_t = cusp::csr_matrix<IndexType,ValueType,MemorySpace>;

	public:
		using size_type = IndexType;
		using value_type = ValueType;

//		using view = ...;
//		using const_view = ...;

		csr_matrix() = default;

		csr_matrix(const size_t num_rows, const size_t num_cols, const size_t num_nnz)
				: parent_t(num_rows, num_cols, num_nnz)
		{}

		template<typename MatrixType>
		csr_matrix(const MatrixType& mat) : parent_t(mat)
		{}

		template<typename MatrixType>
		csr_matrix& operator = (const MatrixType& mat)
		{
			parent_t::operator=(mat);
			return *this;
		}

		void swap(csr_matrix& mat)
		{
			parent_t::swap(mat);
		}

		void resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz)
		{
			parent_t::resize(num_rows, num_cols, num_nnz);
		}

		size_t nrows()
		{
			return parent_t::num_rows;
		}

		size_t ncols()
		{
			return parent_t::num_cols;
		}

		size_t nnz()
		{
			return parent_t::num_entries;
		}

	};

}

#endif //MORPHEUS_CSR_MATRIX_HPP
