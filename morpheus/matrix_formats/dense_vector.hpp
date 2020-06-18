/*****************************************************************************
 *
 *  dense_vector.hpp
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

/*! \file dense_vector.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_DENSE_VECTOR_HPP
#define MORPHEUS_MATRIX_FORMATS_DENSE_VECTOR_HPP

#include <cusp/array1d.h>

namespace morpheus
{
	// Currently using the Cusp Interface
	template<typename ValueType, typename MemorySpace>
	class dense_vector : public cusp::array1d<ValueType, MemorySpace>
	{
	private:
		using parent_t = cusp::array1d<ValueType,MemorySpace>;

	public:
		using size_type = std::size_t;
		using value_type = ValueType;

		using reference = dense_vector&;
//		using view = ...;
//		using const_view = ...;

		dense_vector() = default;

		explicit dense_vector(size_type n)
		: parent_t(n) {}

		explicit dense_vector(size_type n, const value_type &value)
		: parent_t(n, value) {}

		dense_vector(const dense_vector &v)
				: parent_t(v) {}

		dense_vector &operator=(const dense_vector &v)
		{
			parent_t::operator=(v);
			return *this;
		}

		template<typename OtherT, typename OtherMem>
		dense_vector(const dense_vector<OtherT, OtherMem> &v)
				: parent_t(v)
		{}

		template<typename OtherT, typename OtherMem>
		dense_vector::reference operator=(const dense_vector<OtherT, OtherMem> &v);

		template<typename InputIterator>
		dense_vector(InputIterator first, InputIterator last)
		: parent_t(first, last)
		{}

	};

}   // end namespace morpheus

#include <morpheus/matrix_formats/detail/dense_vector.inl>

#endif //MORPHEUS_MATRIX_FORMATS_DENSE_VECTOR_HPP
