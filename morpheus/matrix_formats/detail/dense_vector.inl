/*****************************************************************************
 *
 *  dense_vector.inl
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

/*! \file dense_vector.inl
 *  \brief Description
 */

#ifndef MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_VECTOR_INL
#define MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_VECTOR_INL

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail

	template<typename ValueType, typename MemorySpace>
	template<typename OtherT, typename OtherMem>
	typename dense_vector<ValueType, MemorySpace>::reference
	dense_vector<ValueType, MemorySpace>
    ::operator=(const dense_vector<OtherT, OtherMem> &v)
	{
		parent_t::operator=(v);
		return *this;
	}

}   // end namespace morpheus

#endif //MORPHEUS_MATRIX_FORMATS_DETAIL_DENSE_VECTOR_INL
