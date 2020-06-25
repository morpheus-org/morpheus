/*****************************************************************************
 *
 *  dynamic_matrix.hpp
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

/*! \file dynamic_matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_HPP
#define MORPHEUS_DYNAMIC_MATRIX_HPP

#include <morpheus/matrix.hpp>
#include <morpheus/memory.hpp>

#include <morpheus/matrix_formats/coo_matrix.hpp>
#include <morpheus/matrix_formats/csr_matrix.hpp>
#include <morpheus/matrix_formats/dia_matrix.hpp>
#include <morpheus/matrix_formats/ell_matrix.hpp>
#include <morpheus/matrix_formats/hyb_matrix.hpp>
#include <morpheus/matrix_formats/dense_matrix.hpp>

#include <boost/mpl/vector.hpp>

namespace morpheus
{
	template<typename IndexType, typename ValueType, typename MemorySpace>
	using matrix_t = boost::mpl::vector<morpheus::coo_matrix<IndexType, ValueType, MemorySpace>,
										morpheus::csr_matrix<IndexType, ValueType, MemorySpace>,
										morpheus::dia_matrix<IndexType, ValueType, MemorySpace>,
										morpheus::ell_matrix<IndexType, ValueType, MemorySpace>,
										morpheus::hyb_matrix<IndexType, ValueType, MemorySpace>,
										morpheus::dense_matrix<ValueType, MemorySpace>>;

	template<typename ValueType, typename MemorySpace = morpheus::host_memory>
	using Matrix_i = morpheus::matrix<morpheus::matrix_t<int, ValueType, MemorySpace>>;

	template<typename ValueType, typename MemorySpace = morpheus::host_memory>
	using Matrix_l = morpheus::matrix<morpheus::matrix_t<long, ValueType, MemorySpace>>;

}   // end namespace morpheus

#endif //MORPHEUS_DYNAMIC_MATRIX_HPP
