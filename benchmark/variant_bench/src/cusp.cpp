/*****************************************************************************
 *
 *  cusp.cpp
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

/*! \file cusp.cpp
 *  \brief Description
 */

#include <morpheus/util/parser.hpp>
#include <morpheus/util/timer.hpp>

#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>

int main(int argc, char* argv[])
{
	morpheus::CommandLineParser args;
	args.get(argc, argv).print();

	morpheus::TimerPool timer;

	timer.start(morpheus::TimerPool::timer_id::TOTAL);

	cusp::coo_matrix<int, double, cusp::host_memory> A;

	timer.start(morpheus::TimerPool::timer_id::IO_READ);
	cusp::io::read_matrix_market_file(A, args.fin);
	timer.stop(morpheus::TimerPool::timer_id::IO_READ);

	cusp::array1d<double, cusp::host_memory> x, y(A.num_rows);

	for(int i = 0; i < args.iterations; i++)
	{
		cusp::random_array<double> r(A.num_rows, i);
		x = cusp::array1d<double, cusp::host_memory>(r.begin(), r.end());
		timer.start(morpheus::TimerPool::timer_id::SPMV);
		cusp::multiply(A, x, y);
		timer.stop(morpheus::TimerPool::timer_id::SPMV);
	}

//	timer.start(morpheus::TimerPool::timer_id::IO_WRITE);
//	cusp::io::write_matrix_market_file(x, args.fx);
//	cusp::io::write_matrix_market_file(y, args.fy);
//	timer.stop(morpheus::TimerPool::timer_id::IO_WRITE);

	timer.stop(morpheus::TimerPool::timer_id::TOTAL);

	// Stats
	std::cout << "Matrix Shape\t" << A.num_rows  << "\t" << A.num_cols << "\t" << A.num_entries  << std::endl;
	timer.statistics();

	return 0;
}
