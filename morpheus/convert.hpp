/*****************************************************************************
 *
 *  convert.hpp
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

/*! \file convert.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_CONVERT_HPP
#define MORPHEUS_CONVERT_HPP

#include <morpheus/matrix.hpp>

namespace morpheus
{

	template <typename Types, typename Matrix>
	void convert(matrix<Types> const& src, Matrix & dst);

	template <typename Types, typename Matrix>
	void convert(Matrix const& src, matrix<Types> & dst);

	template <typename Types1, typename Types2>
	void convert(matrix<Types1> const& src, matrix<Types2> & dst);

}   // end namespace morpheus

#include <morpheus/detail/convert.inl>

#endif //MORPHEUS_CONVERT_HPP
