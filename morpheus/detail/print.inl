/*****************************************************************************
 *
 *  print.inl
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

/*! \file print.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DETAIL_PRINT_INL
#define MORPHEUS_DETAIL_PRINT_INL

#include <morpheus/apply_operation.hpp>

#include <morpheus/matrix_formats/print.hpp>

namespace morpheus
{
	namespace detail
	{
		struct print_fn
		{
			using result_type = void;

			template <typename T>
			result_type operator()(const T& mat) const
			{
				morpheus::print(mat);
			}
		};

	}   // end namespace detail

	template <typename Types>
	void print(matrix<Types> const& mat)
	{
		apply_operation(mat, detail::print_fn());
	}

}   // end namespace morpheus

#endif //MORPHEUS_DETAIL_PRINT_INL
