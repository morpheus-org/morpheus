/*****************************************************************************
 *
 *  apply_visitor.inl
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

/*! \file apply_visitor.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_DETAIL_APPLY_VISITOR_INL
#define MORPHEUS_DYNAMIC_MATRIX_DETAIL_APPLY_VISITOR_INL

#include <morpheus/config.hpp>
#include <variant>

namespace morpheus
{
    template <typename Visitor, typename Visitable>
	MORPHEUS_INLINE
	typename Visitor::result_type
	apply_visitor(Visitor& visitor, Visitable&& visitable)
	{
		return std::visit(visitor, visitable);
	}

	template <typename Visitor, typename Visitable>
	MORPHEUS_INLINE
	typename Visitor::result_type
	apply_visitor(const Visitor& visitor, Visitable&& visitable)
	{
		return std::visit(visitor, visitable);
	}

	template <typename Visitor, typename Visitable1, typename Visitable2>
	MORPHEUS_INLINE
	typename Visitor::result_type
	apply_visitor( Visitor& visitor, Visitable1&& visitable1, Visitable2&& visitable2)
	{
		return std::visit(visitor, visitable1, visitable2);
	}

	template <typename Visitor, typename Visitable1, typename Visitable2>
	MORPHEUS_INLINE
	typename Visitor::result_type
	apply_visitor( const Visitor& visitor , Visitable1&& visitable1 , Visitable2&& visitable2)
	{
		return std::visit(visitor, visitable1, visitable2);
	}
}

#endif  //MORPHEUS_DYNAMIC_MATRIX_DETAIL_APPLY_VISITOR_INL