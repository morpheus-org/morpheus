/*****************************************************************************
 *
 *  apply_operation.hpp
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

/*! \file apply_operation.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_APPLY_OPERATION_HPP
#define MORPHEUS_DYNAMIC_MATRIX_APPLY_OPERATION_HPP

namespace morpheus
{

    template <typename Variant, typename UnaryOp>
	auto apply_operation(Variant& arg, UnaryOp op) -> typename UnaryOp::result_type;

	template <typename Variant, typename UnaryOp>
	auto apply_operation(Variant const& arg, UnaryOp op) -> typename UnaryOp::result_type;

	template <typename Variant1, typename Variant2, typename BinaryOp>
	auto apply_operation(
			Variant1 const& arg1,
			Variant2 const& arg2,
			BinaryOp op) -> typename BinaryOp::result_type;

	template <typename Variant1, typename Variant2, typename BinaryOp>
	auto apply_operation(
			Variant1 const& arg1,
			Variant2 & arg2,
			BinaryOp op) -> typename BinaryOp::result_type;

	template <typename Variant1, typename Variant2, typename BinaryOp>
	auto apply_operation(
			Variant1 & arg1,
			Variant2 & arg2,
			BinaryOp op) -> typename BinaryOp::result_type;

}   // end namespace morpheus

#include <morpheus/dynamic_matrix/detail/apply_operation.inl>

#endif  //MORPHEUS_DYNAMIC_MATRIX_APPLY_OPERATION_HPP