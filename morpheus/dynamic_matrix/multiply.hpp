/*****************************************************************************
 *
 *  multiply.hpp
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

/*! \file multiply.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_MULTIPLY_HPP
#define MORPHEUS_DYNAMIC_MATRIX_MULTIPLY_HPP

#include <morpheus/dynamic_matrix/matrix.hpp>

namespace morpheus
{

	// template <typename DerivedPolicy,
	// 		typename VariantFormats,
	// 		typename Vector1,
	// 		typename Vector2>
	// void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
	//               matrix<VariantFormats> const& A,
	//               Vector1 const& B,
	//               Vector2 &C);

	template <typename VariantFormats, typename Vector1, typename Vector2>
	void multiply(matrix<VariantFormats> const& A,
				  Vector1 const& B,
				  Vector2 &C);

}   // end namespace morpheus

#include <morpheus/dynamic_matrix/detail/multiply.inl>

#endif //MORPHEUS_DYNAMIC_MATRIX_MULTIPLY_HPP
