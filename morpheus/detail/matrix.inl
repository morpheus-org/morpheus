/*****************************************************************************
 *
 *  matrix.inl
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

/*! \file matrix.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DETAIL_MATRIX_INL
#define MORPHEUS_DETAIL_MATRIX_INL

#include <morpheus/apply_operation.hpp>

namespace morpheus
{
	namespace detail
	{
		struct any_type_get_num_rows
		{
			using result_type = size_t;
			template <typename T>
			result_type operator()(T& v) const { return v.nrows(); }
			// TODO:: const T& v fails for some reason
		};

		struct any_type_get_num_cols
		{
			using result_type = size_t;
			template <typename T>
			result_type operator()(T& v) const { return v.ncols(); }
			// TODO:: const T& v fails for some reason
		};

		struct any_type_get_num_nnz
		{
			using result_type = size_t;
			template <typename T>
			result_type operator()(T& v) const { return v.nnz(); }
			// TODO:: const T& v fails for some reason
		};
	}   // end namespace detail

	template<typename Matrices>
	typename matrix<Matrices>::reference
    matrix<Matrices>::operator=(matrix const& mat)
	{
		parent_t::operator=((parent_t const&)mat);
		return *this;
	}

	template<typename Matrices>
	template<typename Matrix>
	typename matrix<Matrices>::reference
	matrix<Matrices>::operator= (const Matrix &mat)
	{
		parent_t::operator=(mat);
		return *this;
	}

	template<typename Matrices>
	template<typename OtherMatrices>
	typename matrix<Matrices>::reference
	matrix<Matrices>::operator= (const matrix<OtherMatrices> &mat)
	{
		parent_t::operator=((typename make_variant_over<OtherMatrices>::type const&)mat);
		return *this;
	}

	template<typename Matrices>
	size_t matrix<Matrices>::nrows()
	{
		return apply_operation(*this, detail::any_type_get_num_rows());
	}

	template<typename Matrices>
	size_t matrix<Matrices>::ncols()
	{
		return apply_operation(*this, detail::any_type_get_num_cols());
	}

	template<typename Matrices>
	size_t matrix<Matrices>::nnz()
	{
		return apply_operation(*this, detail::any_type_get_num_nnz());
	}

}   // end namespace morpheus

#endif //MORPHEUS_DETAIL_MATRIX_INL
