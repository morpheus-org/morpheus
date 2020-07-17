/*****************************************************************************
 *
 *  dynamic_1.cpp
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

/*! \file dynamic_1.cpp
 *  \brief Description
 */

#include <morpheus/util/parser.hpp>
#include <morpheus/util/timer.hpp>

#include <morpheus/dynamic_matrix.hpp>
#include <morpheus/matrix_formats/dense_vector.hpp>
#include <morpheus/io/matrix_market.hpp>
#include <morpheus/multiply.hpp>

#include <benchmark/variant_active_front/src/dynamic.hpp>

int main(int argc, char* argv[])
{
	morpheus::CommandLineParser args;
	args.get(argc, argv).print();

	morpheus::TimerPool timer;

	timer.start(morpheus::TimerPool::timer_id::TOTAL);

	Matrix A;
	A = morpheus::coo_matrix<int, double, morpheus::host_memory>();

	timer.start(morpheus::TimerPool::timer_id::IO_READ);
	morpheus::io::read_matrix_market_file(A, args.fin);
	timer.stop(morpheus::TimerPool::timer_id::IO_READ);

	morpheus::dense_vector<double, morpheus::host_memory> x, y(A.nrows());

	timer.start(morpheus::TimerPool::timer_id::SPMV);
	for(int i = 0; i < args.iterations; i++)
	{
		cusp::random_array<double> r(A.nrows(), i);
		x = morpheus::dense_vector<double, morpheus::host_memory>(r.begin(), r.end());
		morpheus::multiply(A, x, y);
	}
	timer.stop(morpheus::TimerPool::timer_id::SPMV);

	timer.stop(morpheus::TimerPool::timer_id::TOTAL);

	// Stats
	std::cout << "Matrix Shape\t" <<  A.nrows() << "\t" << A.ncols() << "\t" << A.nnz()  << std::endl;
	timer.statistics();

	return 0;
}