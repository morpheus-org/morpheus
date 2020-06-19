/*****************************************************************************
 *
 *  cusp.hpp
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

/*! \file cusp.hpp
 *  \brief Description
 */

#ifndef EXAMPLES_INCLUDE_CUSP_HPP
#define EXAMPLES_INCLUDE_CUSP_HPP

#include <cusp/io/matrix_market.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/multiply.h>

// Cusp stuff
using IndexType = int;
using ValueType = double;
using host = cusp::host_memory;

using Coo_matrix = cusp::coo_matrix<IndexType, ValueType, host>;
using Csr_matrix = cusp::csr_matrix<IndexType, ValueType, host>;
using Dense_matrix = cusp::array2d<ValueType, host>;
using Dense_vector = cusp::array1d<ValueType, host>;
using Random_vector = cusp::random_array<ValueType>;

#endif //EXAMPLES_INCLUDE_CUSP_HPP
