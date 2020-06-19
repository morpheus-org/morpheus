/*****************************************************************************
 *
 *  static.hpp
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

/*! \file static.hpp
 *  \brief Description
 */

#ifndef EXAMPLES_INCLUDE_STATIC_HPP
#define EXAMPLES_INCLUDE_STATIC_HPP

#include <morpheus/matrix_formats/io/matrix_market.hpp>
#include <morpheus/matrix_formats/coo_matrix.hpp>
#include <morpheus/matrix_formats/csr_matrix.hpp>
#include <morpheus/matrix_formats/dense_matrix.hpp>
#include <morpheus/matrix_formats/dense_vector.hpp>
#include <morpheus/matrix_formats/multiply.hpp>
#include <morpheus/memory.hpp>

using IndexType = int;
using ValueType = double;
using host = morpheus::host_memory;

using Coo_matrix = morpheus::coo_matrix<IndexType, ValueType, host>;
using Csr_matrix = morpheus::csr_matrix<IndexType, ValueType, host>;
using Dense_matrix = morpheus::dense_matrix<ValueType, host>;
using Dense_vector = morpheus::dense_vector<ValueType, host>;
using Random_vector = cusp::random_array<ValueType>;

#endif //EXAMPLES_INCLUDE_STATIC_HPP
