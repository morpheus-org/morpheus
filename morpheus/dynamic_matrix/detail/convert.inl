/*****************************************************************************
 *
 *  convert.inl
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

/*! \file convert.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_DETAIL_CONVERT_INL
#define MORPHEUS_DYNAMIC_MATRIX_DETAIL_CONVERT_INL

#include <morpheus/config.hpp>
#include <morpheus/matrix_formats/convert.hpp>

namespace morpheus
{
	namespace detail
	{
		struct convert_fn : binary_operation_obj<convert_fn>
		{
			template <typename T1, typename T2>
			MORPHEUS_INLINE
			void apply_compatible(T1 const& src, T2& dst) const
			{
				morpheus::convert(src, dst);
			}
		};

	}   // end namespace detail

    template <typename VariantFormats1, typename Matrix>
	MORPHEUS_INLINE
	void convert(matrix<VariantFormats1> const& src, Matrix & dst)
	{
		apply_operation(src.types(), std::bind(detail::convert_fn(), std::placeholders::_1, std::ref(dst)));
	}

	template <typename VariantFormats1, typename Matrix>
	MORPHEUS_INLINE
	void convert(Matrix const& src, matrix<VariantFormats1> & dst)
	{
		apply_operation(dst.types(), std::bind(detail::convert_fn(), std::cref(src), std::placeholders::_1));
	}

	template <typename VariantFormats1, typename VariantFormats2>
	MORPHEUS_INLINE
	void convert(matrix<VariantFormats1> const& src, matrix<VariantFormats2> & dst)
	{
		apply_operation(src.types(), dst.types(), detail::convert_fn());
	}

}   // end namespace morpheus

#endif  //MORPHEUS_DYNAMIC_MATRIX_DETAIL_CONVERT_INL