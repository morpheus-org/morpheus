/*****************************************************************************
 *
 *  dynamic.hpp
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

/*! \file dynamic.hpp
 *  \brief Description
 */

#ifndef EXAMPLES_DYNAMIC_HPP
#define EXAMPLES_DYNAMIC_HPP

#include <morpheus/matrix.hpp>
#include <morpheus/io/matrix_market.hpp>
#include <morpheus/multiply.hpp>
#include <morpheus/convert.hpp>

#include <boost/mpl/vector.hpp>

#include <examples/include/static.hpp>

// Dynamic matrix
using matrix_t = boost::mpl::vector<Coo_matrix, Csr_matrix, Dense_matrix>;
using Matrix = morpheus::matrix<matrix_t>;

#endif //EXAMPLES_DYNAMIC_HPP
