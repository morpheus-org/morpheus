/*****************************************************************************
 *
 *  apply_operation.inl
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

/*! \file apply_operation.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DETAIL_APPLY_OPERATION_INL
#define MORPHEUS_DETAIL_APPLY_OPERATION_INL

#include <morpheus/apply_visitor.hpp>

namespace morpheus
{
	namespace detail
	{
		// Needed for when two variants will be used together
		template <typename Derived, typename Result=void>
		struct binary_operation_obj
		{
			using result_type = Result;

			template <typename V1, typename V2>
			result_type operator()(const V1& v1,const V2& v2) const {
				return apply(v1, v2);
			}

			template <typename V1, typename V2>
			result_type operator()(const V1& v1, V2& v2) const {
				return apply(v1, v2);
			}

			template <typename V1, typename V2>
			result_type operator()(V1& v1, V2& v2) const {
				return apply(v1, v2);
			}

		private:
			// dispatch from apply overload to a function with distinct name
			template <typename V1, typename V2>
			result_type apply(V1 const& v1, V2 const& v2) const
			{
				return ((const Derived*)this)->apply_compatible(v1, v2);
			}

			template <typename V1, typename V2>
			result_type apply(V1 const& v1, V2 & v2) const
			{
				return ((const Derived*)this)->apply_compatible(v1, v2);
			}

			template <typename V1, typename V2>
			result_type apply(V1& v1, V2& v2) const
			{
				return ((const Derived*)this)->apply_compatible(v1, v2);
			}
		};

	}   // end namespace detail

	template <typename Types, typename UnaryOp>
	auto apply_operation(variant<Types>& arg, UnaryOp op) -> typename UnaryOp::result_type
	{
		return ::morpheus::apply_visitor(op, arg);
	}

	template <typename Types, typename UnaryOp>
	auto apply_operation(variant<Types> const& arg, UnaryOp op) -> typename UnaryOp::result_type
	{
		return ::morpheus::apply_visitor(op, arg);
	}

	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			variant<Types1> const& arg1,
			variant<Types2> const& arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return ::morpheus::apply_visitor(op, arg1, arg2);
	}

	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			variant<Types1> const& arg1,
			variant<Types2> & arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return ::morpheus::apply_visitor(op, arg1, arg2);
	}

	template <typename Types1, typename Types2, typename BinaryOp>
	auto apply_operation(
			boost::variant<Types1> & arg1,
			boost::variant<Types2> & arg2,
			BinaryOp op) -> typename BinaryOp::result_type
	{
		return ::morpheus::apply_visitor(op, arg1, arg2);
	}

}   // end namespace morpheus

#endif //MORPHEUS_DETAIL_APPLY_OPERATION_INL
