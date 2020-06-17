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

#ifndef MORPHEUS_DETAIL_CONVERT_INL
#define MORPHEUS_DETAIL_CONVERT_INL

#include <morpheus/apply_operation.hpp>

#include <morpheus/matrix_formats/convert.hpp>

namespace morpheus
{
	namespace detail
	{
		struct convert_fn : binary_operation_obj<convert_fn>
		{
			template <typename T1, typename T2>
			void apply_compatible(T1 const& src, T2& dst) const
			{
				morpheus::convert(src, dst);
			}
		};

	}   // end namespace detail

	template <typename Types, typename Matrix>
	void convert(matrix<Types> const& src, Matrix & dst)
	{
		apply_operation(src, std::bind(detail::convert_fn(), std::placeholders::_1, std::ref(dst)));
	}

	template <typename Types, typename Matrix>
	void convert(Matrix const& src, matrix<Types> & dst)
	{
		apply_operation(dst, std::bind(detail::convert_fn(), std::cref(src), std::placeholders::_1));
	}

	template <typename Types1, typename Types2>
	void convert(matrix<Types1> const& src, matrix<Types2> & dst)
	{
		apply_operation(src, dst, detail::convert_fn());
	}

}   // end namespace morpheus

#endif //MORPHEUS_DETAIL_CONVERT_INL
