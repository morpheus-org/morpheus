/*****************************************************************************
 *
 *  ell_matrix.hpp
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

/*! \file ell_matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_ELL_MATRIX_HPP
#define MORPHEUS_MATRIX_FORMATS_ELL_MATRIX_HPP

#include <cusp/ell_matrix.h>

namespace morpheus
{
	// Currently using the Cusp Interface
	template <typename IndexType, typename ValueType, class MemorySpace>
	class ell_matrix : public cusp::ell_matrix<IndexType,ValueType,MemorySpace>
	{
	private:
		using parent_t = cusp::ell_matrix<IndexType,ValueType,MemorySpace>;

	public:
		using size_type = IndexType;
		using value_type = ValueType;

		using reference = ell_matrix&;

		ell_matrix() = default;

		ell_matrix(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
		           const size_t num_entries_per_row, const size_t alignment = 32);

		template<typename MatrixType>
		ell_matrix(const MatrixType& matrix);

		template<typename MatrixType>
		ell_matrix::reference operator = (const MatrixType& matrix);

		void resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
		            const size_t num_entries_per_row);

		void resize(const size_t num_rows, const size_t num_cols, const size_t num_nnz,
		            const size_t num_entries_per_row, const size_t alignment);

		void swap(ell_matrix& matrix);

		size_t nrows();

		size_t ncols();

		size_t nnz();

	};

}   // end namespace morpheus

#include <morpheus/matrix_formats/detail/ell_matrix.inl>

#endif //MORPHEUS_MATRIX_FORMATS_ELL_MATRIX_HPP