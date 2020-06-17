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

#ifndef MORPHEUS_MATRIX_FORMATS_MULTIPLY_HPP
#define MORPHEUS_MATRIX_FORMATS_MULTIPLY_HPP

#include <thrust/execution_policy.h>

namespace morpheus
{
	template <typename DerivedPolicy,
			typename LinearOperator,
			typename MatrixOrVector1,
			typename MatrixOrVector2>
	void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
	              const LinearOperator&  A,
	              const MatrixOrVector1& B,
	              MatrixOrVector2& C);

	template <typename LinearOperator,
			typename MatrixOrVector1,
			typename MatrixOrVector2>
	void multiply(const LinearOperator&  A,
	              const MatrixOrVector1& B,
	              MatrixOrVector2& C);

}   // end namespace morpheus

#include <morpheus/matrix_formats/detail/multiply.inl>

#endif //MORPHEUS_MATRIX_FORMATS_MULTIPLY_HPP
