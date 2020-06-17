/*****************************************************************************
 *
 *  coo_matrix.hpp
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

/*! \file coo_matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_COO_MATRIX_HPP
#define MORPHEUS_MATRIX_FORMATS_COO_MATRIX_HPP

#include <cusp/coo_matrix.h>

namespace morpheus
{
	// Currently using the Cusp Interface
	template <typename IndexType, typename ValueType, class MemorySpace>
	class coo_matrix : public cusp::coo_matrix<IndexType,ValueType,MemorySpace>
	{
	private:
		using parent_t = cusp::coo_matrix<IndexType,ValueType,MemorySpace>;

	public:
		using size_type = IndexType;
		using value_type = ValueType;

		using reference = coo_matrix&;

//		using view = ...;
//		using const_view = ...;

		coo_matrix() = default;

		coo_matrix(const size_t num_rows, const size_t num_cols, const size_t num_nnz)
				: parent_t(num_rows, num_cols, num_nnz)
		{}

		template<typename MatrixType>
		coo_matrix(const MatrixType& mat) : parent_t(mat)
		{}

		template<typename MatrixType>
		coo_matrix::reference operator = (const MatrixType& mat);

		void swap(coo_matrix& mat);

		void resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz);

		size_t nrows();

		size_t ncols();

		size_t nnz();

	};

}   // end namespace morpheus

#include <morpheus/matrix_formats/detail/coo_matrix.inl>

#endif //MORPHEUS_MATRIX_FORMATS_COO_MATRIX_HPP
