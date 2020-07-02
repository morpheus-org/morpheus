/*****************************************************************************
 *
 *  dynamic_selection.cpp
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

/*! \file dynamic_selection.cpp
 *  \brief Description
 */

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

#include <iostream>

#include <morpheus/util/parser.hpp>
#include <morpheus/util/timer.hpp>

#include <morpheus/dynamic_matrix.hpp>
#include <morpheus/format_selector.hpp>
#include <morpheus/matrix_formats/dense_vector.hpp>
#include <morpheus/io/matrix_market.hpp>
#include <morpheus/multiply.hpp>
#include <morpheus/convert.hpp>

using mat_t = boost::mpl::vector<morpheus::coo_matrix<int, double, morpheus::host_memory>,
								 morpheus::csr_matrix<int, double, morpheus::host_memory>,
								 morpheus::dia_matrix<int, double, morpheus::host_memory>,
								 morpheus::ell_matrix<int, double, morpheus::host_memory>,
								 morpheus::hyb_matrix<int, double, morpheus::host_memory>,
								 morpheus::dense_matrix<double, morpheus::host_memory>>;

using Matrix = morpheus::matrix<mat_t>;

int main(int argc, char* argv[])
{
	morpheus::CommandLineParser args;
	args.get(argc, argv).print();

	int format = morpheus::FMT_COO;  // Default

	if(args.format == 1){
		format = morpheus::FMT_CSR;
	}else if(args.format == 2){
		format = morpheus::FMT_DIA;
	}else if(args.format == 3){
		format = morpheus::FMT_ELL;
	}else if(args.format == 4){
		format = morpheus::FMT_HYB;
	}else if(args.format == 5){
		format = morpheus::FMT_DENSE;
	}else{
		std::cerr << "Invalid matrix storage format.";
		exit(-1);
	}

	morpheus::TimerPool timer;

	timer.start(morpheus::TimerPool::timer_id::TOTAL);

	Matrix A, B;

	A = morpheus::coo_matrix<int, double, morpheus::host_memory>();
	morpheus::select_format(format, B);

	timer.start(morpheus::TimerPool::timer_id::IO_READ);
	morpheus::io::read_matrix_market_file(A, args.fin);
	timer.stop(morpheus::TimerPool::timer_id::IO_READ);

	timer.start(morpheus::TimerPool::timer_id::CONVERT);
	morpheus::convert(A, B);
	timer.stop(morpheus::TimerPool::timer_id::CONVERT);

	morpheus::dense_vector<double, morpheus::host_memory> x, y(A.nrows());

	for(int i = 0; i < args.iterations; i++)
	{
		cusp::random_array<double> r(A.nrows(), i);
		x = morpheus::dense_vector<double, morpheus::host_memory>(r.begin(), r.end());
		timer.start(morpheus::TimerPool::timer_id::SPMV);
		morpheus::multiply(A, x, y);
		timer.stop(morpheus::TimerPool::timer_id::SPMV);
	}

//	timer.start(morpheus::TimerPool::timer_id::IO_WRITE);
//	morpheus::io::write_matrix_market_file(x, args.fx);
//	morpheus::io::write_matrix_market_file(y, args.fy);
//	timer.stop(morpheus::TimerPool::timer_id::IO_WRITE);

	timer.stop(morpheus::TimerPool::timer_id::TOTAL);

	// Stats
	std::cout << "Matrix Shape\t" <<  A.nrows() << "\t" << A.ncols() << "\t" << A.nnz()  << std::endl;
	timer.statistics();

	return 0;
}