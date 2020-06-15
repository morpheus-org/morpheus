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

#ifndef MORPHEUS_DYNAMIC_APPLY_OPERATION_HPP
#define MORPHEUS_DYNAMIC_APPLY_OPERATION_HPP

#include <boost/variant/apply_visitor.hpp>

namespace morpheus
{

	/// Invokes a generic mutable operation (represented as a unary function object) on a variant
	template <typename Types, typename UnaryOp>
	auto apply_operation(boost::variant<Types>& arg, UnaryOp op) -> typename UnaryOp::result_type
	{
		return apply_visitor(op, arg);
	}

	/// Invokes a generic constant operation (represented as a unary function object) on a variant
	template <typename Types, typename UnaryOp>
	auto apply_operation(boost::variant<Types> const& arg, UnaryOp op) -> typename UnaryOp::result_type
	{
		return apply_visitor(op, arg);
	}

	/// Invokes a generic constant operation (represented as a binary function object) on two variants
	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			boost::variant<Types1> const& arg1,
			boost::variant<Types2> const& arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return apply_visitor(op, arg1, arg2);
	}

	/// Invokes a generic mutable operation (represented as a binary function object) on the second variant
	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			boost::variant<Types1> const& arg1,
			boost::variant<Types2> & arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return apply_visitor(op, arg1, arg2);
	}

	/// Invokes a generic mutable operation (represented as a binary function object) on both variants
	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			boost::variant<Types1> & arg1,
			boost::variant<Types2> & arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return apply_visitor(op, arg1, arg2);
	}

}

#endif //MORPHEUS_DYNAMIC_APPLY_OPERATION_HPP
