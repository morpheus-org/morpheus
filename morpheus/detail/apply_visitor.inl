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

#ifndef MORPHEUS_DETAIL_APPLY_VISITOR_INL
#define MORPHEUS_DETAIL_APPLY_VISITOR_INL

#include <boost/variant/apply_visitor.hpp>

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail

	template <typename Visitor, typename Visitable>
	inline typename Visitor::result_type
	apply_visitor(Visitor& visitor, Visitable&& visitable)
	{
		return boost::apply_visitor(visitor, visitable);
	}

	template <typename Visitor, typename Visitable>
	inline typename Visitor::result_type
	apply_visitor(const Visitor& visitor, Visitable&& visitable)
	{
		return boost::apply_visitor(visitor, visitable);
	}

	template <typename Visitor, typename Visitable1, typename Visitable2>
	inline typename Visitor::result_type
	apply_visitor( Visitor& visitor, Visitable1&& visitable1, Visitable2&& visitable2)
	{
		return boost::apply_visitor(visitor, visitable1, visitable2);
	}

	template <typename Visitor, typename Visitable1, typename Visitable2>
	inline typename Visitor::result_type
	apply_visitor( const Visitor& visitor , Visitable1&& visitable1 , Visitable2&& visitable2)
	{
		return boost::apply_visitor(visitor, visitable1, visitable2);
	}

	/// TODO:: Fallback in case move semantics not supported
//	template <typename Visitor, typename Visitable>
//	inline typename Visitor::result_type
//	apply_visitor(Visitor& visitor, Visitable& visitable)
//	{
//		return boost::apply_visitor(visitor, visitable);
//	}
//
//	template <typename Visitor, typename Visitable>
//	inline typename Visitor::result_type
//	apply_visitor(const Visitor& visitor, Visitable& visitable)
//	{
//		return boost::apply_visitor(visitor, visitable);
//	}
//
//	template <typename Visitor, typename Visitable1, typename Visitable2>
//	inline typename Visitor::result_type
//	apply_visitor( Visitor& visitor, Visitable1& visitable1, Visitable2& visitable2)
//	{
//		return boost::apply_visitor(visitor, visitable1, visitable2);
//	}
//
//	template <typename Visitor, typename Visitable1, typename Visitable2>
//	inline typename Visitor::result_type
//	apply_visitor( const Visitor& visitor , Visitable1& visitable1 , Visitable2& visitable2)
//	{
//		return boost::apply_visitor(visitor, visitable1, visitable2);
//	}

}   // end namespace morpheus

#endif //MORPHEUS_DETAIL_APPLY_VISITOR_INL
