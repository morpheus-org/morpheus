/*****************************************************************************
 *
 *  coo.cpp
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

/*! \file coo.cpp
 *  \brief Description
 */

#include <examples/include/parser.hpp>
#include <examples/include/timer.hpp>
#include <examples/include/static.hpp>

using namespace morpheus::examples;

int main(int argc, char* argv[])
{
	parser args;
	args.get(argc, argv).print();

	timer total("Static_Total"), reader("Static_Reader"), writer("Static_Writer") , spmv("Static_spMv");

	total.start();

	Coo_matrix A;

	reader.start();

	morpheus::io::read_matrix_market_file(A, args.fin);

	reader.stop();

	Dense_vector y(A.nrows());
	Dense_vector x;

	for(int i = 0; i < args.iterations; i++)
	{
		Random_vector r(A.nrows(), i);
		x = Dense_vector(r.begin(), r.end());
		spmv.start();
		morpheus::multiply(A, x, y);
		spmv.stop();
	}

	writer.start();
	morpheus::io::write_matrix_market_file(x, args.fx);
	morpheus::io::write_matrix_market_file(y, args.fy);
	writer.stop();

	total.stop();

	// Stats
	std::cout << args.filename << "::\tMatrix Shape\t" << A.nrows()  << "\t" << A.ncols() << "\t" << A.nnz()  << std::endl;
	std::cout << args.filename << "::\t" << total << std::endl;
	std::cout << args.filename << "::\t" << reader << std::endl;
	std::cout << args.filename << "::\t" << writer << std::endl;
	std::cout << args.filename << "::\t" << spmv << std::endl;

	return 0;
}

