/*****************************************************************************
 *
 *  matrix.hpp
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

/*! \file matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_HPP
#define MORPHEUS_MATRIX_HPP

#include <morpheus/variant.hpp>

namespace morpheus
{
	// TODO:: Allow change of type from a function using an index

	template<typename Matrices>
	class matrix : public make_variant_over<Matrices>::type
	{
		using parent_t = typename make_variant_over<Matrices>::type;

	public:

		using reference = matrix&;

		matrix() = default;
		matrix(matrix const& mat) : parent_t((parent_t const&) mat)
		{}

		template <typename Matrix>
		explicit matrix(Matrix const& mat) : parent_t(mat)
		{}

		template <typename OtherMatrices>
		matrix(matrix<OtherMatrices> const& mat)
				: parent_t((typename make_variant_over<OtherMatrices>::type const&)mat)
		{}

		matrix& operator=(matrix const& mat);

		template <typename Matrix>
		matrix& operator=(Matrix const& mat);

		template <typename OtherMatrices>
		matrix& operator=(matrix<OtherMatrices> const& mat);

		size_t nrows();

		size_t ncols();

		size_t nnz();
	};

}   // end namespace morpheus

#include <morpheus/detail/matrix.inl>

#endif //MORPHEUS_MATRIX_HPP
