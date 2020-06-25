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

#ifndef BENCHMARK_VARIANT_BENCH_SRC_DYNAMIC_HPP
#define BENCHMARK_VARIANT_BENCH_SRC_DYNAMIC_HPP

#ifndef NUM_TYPES
#define NUM_TYPES 1
#endif

#include <morpheus/dynamic_matrix.hpp>

#if NUM_TYPES == 1

using mat_t = boost::mpl::vector<morpheus::coo_matrix<int, double, morpheus::host_memory>>;

#elif NUM_TYPES == 6

using mat_t = boost::mpl::vector<morpheus::coo_matrix<int, double, morpheus::host_memory>,
								 morpheus::csr_matrix<int, double, morpheus::host_memory>,
								 morpheus::dia_matrix<int, double, morpheus::host_memory>,
								 morpheus::ell_matrix<int, double, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, double, morpheus::host_memory>,
								 morpheus::dense_matrix<double, morpheus::host_memory>>;

#elif NUM_TYPES == 12

using mat_t = boost::mpl::vector<morpheus::coo_matrix<int, double, morpheus::host_memory>,
								 morpheus::csr_matrix<int, double, morpheus::host_memory>,
								 morpheus::dia_matrix<int, double, morpheus::host_memory>,
								 morpheus::ell_matrix<int, double, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, double, morpheus::host_memory>,
								 morpheus::coo_matrix<int, float, morpheus::host_memory>,
								 morpheus::csr_matrix<int, float, morpheus::host_memory>,
								 morpheus::dia_matrix<int, float, morpheus::host_memory>,
								 morpheus::ell_matrix<int, float, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, float, morpheus::host_memory>,
								 morpheus::dense_matrix<double, morpheus::host_memory>,
								 morpheus::dense_matrix<float, morpheus::host_memory>>;

#elif NUM_TYPES == 20

using mat_t = boost::mpl::vector<morpheus::coo_matrix<int, double, morpheus::host_memory>,
								 morpheus::csr_matrix<int, double, morpheus::host_memory>,
								 morpheus::dia_matrix<int, double, morpheus::host_memory>,
								 morpheus::ell_matrix<int, double, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, double, morpheus::host_memory>,
								 morpheus::coo_matrix<int, float, morpheus::host_memory>,
								 morpheus::csr_matrix<int, float, morpheus::host_memory>,
								 morpheus::dia_matrix<int, float, morpheus::host_memory>,
								 morpheus::ell_matrix<int, float, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, float, morpheus::host_memory>,
								 morpheus::coo_matrix<long long, double, morpheus::host_memory>,
								 morpheus::csr_matrix<long long, double, morpheus::host_memory>,
								 morpheus::dia_matrix<long long, double, morpheus::host_memory>,
								 morpheus::ell_matrix<long long, double, morpheus::host_memory>,
								 morpheus::hyb_matrix<long long, double, morpheus::host_memory>,
								 morpheus::coo_matrix<long long, float, morpheus::host_memory>,
								 morpheus::csr_matrix<long long, float, morpheus::host_memory>,
								 morpheus::dia_matrix<long long, float, morpheus::host_memory>,
								 morpheus::ell_matrix<long long, float, morpheus::host_memory>,
								 morpheus::hyb_matrix<long long, float, morpheus::host_memory>>;

#endif

using Matrix = morpheus::matrix<mat_t>;

#endif //BENCHMARK_VARIANT_BENCH_SRC_DYNAMIC_HPP
