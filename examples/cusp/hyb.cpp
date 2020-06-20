/*****************************************************************************
 *
 *  hyb.cpp
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

/*! \file hyb.cpp
 *  \brief Description
 */

#include <examples/include/parser.hpp>
#include <examples/include/timer.hpp>

#include <cusp/hyb_matrix.h>
#include <cusp/array1d.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>

using namespace morpheus::examples;

int main(int argc, char* argv[])
{
	parser args;
	args.get(argc, argv).print();

	timer total("Cusp_Total"), reader("Cusp_Reader"), writer("Cusp_Writer") , spmv("Cusp_spMv");

	total.start();

	cusp::hyb_matrix<int, double, cusp::host_memory> A;

	reader.start();

	cusp::io::read_matrix_market_file(A, args.fin);

	reader.stop();

	cusp::array1d<double, cusp::host_memory> x, y(A.num_rows);

	for(int i = 0; i < args.iterations; i++)
	{
		cusp::random_array<double> r(A.num_rows, i);
		x = cusp::array1d<double, cusp::host_memory>(r.begin(), r.end());
		spmv.start();
		cusp::multiply(A, x, y);
		spmv.stop();
	}

	writer.start();
	cusp::io::write_matrix_market_file(x, args.fx);
	cusp::io::write_matrix_market_file(y, args.fy);
	writer.stop();

	total.stop();

	// Stats
	std::cout << args.filename << "::\tMatrix Shape\t" << A.num_rows  << "\t" << A.num_cols << "\t" << A.ell.num_entries + A.coo.num_entries  << std::endl;
	std::cout << args.filename << "::\t" << total << std::endl;
	std::cout << args.filename << "::\t" << reader << std::endl;
	std::cout << args.filename << "::\t" << writer << std::endl;
	std::cout << args.filename << "::\t" << spmv << std::endl;

	return 0;
}