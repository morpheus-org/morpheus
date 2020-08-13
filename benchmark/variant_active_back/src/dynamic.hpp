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

#include <morpheus/dynamic_matrix/matrix.hpp>

#include <morpheus/matrix_formats/coo_matrix.hpp>
#include <morpheus/matrix_formats/csr_matrix.hpp>
#include <morpheus/matrix_formats/ell_matrix.hpp>
#include <morpheus/matrix_formats/dia_matrix.hpp>
#include <morpheus/matrix_formats/hyb_matrix.hpp>
#include <morpheus/matrix_formats/dense_matrix.hpp>

#include <morpheus/memory.hpp>

using Coo_id = morpheus::coo_matrix<int, double, morpheus::host_memory>;
using Csr_id = morpheus::csr_matrix<int, double, morpheus::host_memory>;
using Dia_id = morpheus::dia_matrix<int, double, morpheus::host_memory>;
using Ell_id = morpheus::ell_matrix<int, double, morpheus::host_memory>;
using Hyb_id = morpheus::hyb_matrix<int, double, morpheus::host_memory>;
using Dense_id = morpheus::dense_matrix<double, morpheus::host_memory>;
using Coo_if = morpheus::coo_matrix<int, float, morpheus::host_memory>;
using Csr_if = morpheus::csr_matrix<int, float, morpheus::host_memory>;
using Dia_if = morpheus::dia_matrix<int, float, morpheus::host_memory>;
using Ell_if = morpheus::ell_matrix<int, float, morpheus::host_memory>;
using Hyb_if = morpheus::hyb_matrix<int, float, morpheus::host_memory>;
using Dense_if = morpheus::dense_matrix<float, morpheus::host_memory>;
using Coo_ld = morpheus::coo_matrix<long long, double, morpheus::host_memory>;
using Csr_ld = morpheus::csr_matrix<long long, double, morpheus::host_memory>;
using Dia_ld = morpheus::dia_matrix<long long, double, morpheus::host_memory>;
using Ell_ld = morpheus::ell_matrix<long long, double, morpheus::host_memory>;
using Hyb_ld = morpheus::hyb_matrix<long long, double, morpheus::host_memory>;
using Coo_lf = morpheus::coo_matrix<long long, float, morpheus::host_memory>;
using Csr_lf = morpheus::csr_matrix<long long, float, morpheus::host_memory>;
using Dia_lf = morpheus::dia_matrix<long long, float, morpheus::host_memory>;
using Ell_lf = morpheus::ell_matrix<long long, float, morpheus::host_memory>;
using Hyb_lf = morpheus::hyb_matrix<long long, float, morpheus::host_memory>;

#ifdef BOOST_VARIANT
	#include <boost/variant/variant.hpp>
	#if NUM_TYPES == 1
		using format = boost::variant<Coo_id>;
	#elif NUM_TYPES == 6
		using format = boost::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_id>;
	#elif NUM_TYPES == 12
		using format = boost::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_if,
									  Csr_if,Dia_if,Ell_if,Hyb_if,Dense_if,Coo_id>;
	#elif NUM_TYPES == 20
		using format = boost::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_if,
									Csr_if,Dia_if,Ell_if,Hyb_if,Dense_if,
									Coo_ld,Csr_ld,Dia_ld,Ell_ld,Hyb_ld,
									Coo_lf,Csr_lf,Dia_lf,Coo_id>;
	#endif
#else
	#include <variant>
	#if NUM_TYPES == 1
		using format = std::variant<Coo_id>;
	#elif NUM_TYPES == 6
		using format = std::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_id>;
	#elif NUM_TYPES == 12
		using format = std::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_if,
									  Csr_if,Dia_if,Ell_if,Hyb_if,Dense_if,Coo_id>;
	#elif NUM_TYPES == 20
		using format = std::variant<Csr_id,Dia_id,Ell_id,Hyb_id,Dense_id,Coo_if,
									Csr_if,Dia_if,Ell_if,Hyb_if,Dense_if,
									Coo_ld,Csr_ld,Dia_ld,Ell_ld,Hyb_ld,
									Coo_lf,Csr_lf,Dia_lf,Coo_id>;
	#endif
#endif

using Matrix = morpheus::matrix<format>;

#endif //BENCHMARK_VARIANT_BENCH_SRC_DYNAMIC_HPP
