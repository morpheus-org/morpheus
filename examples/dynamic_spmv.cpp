/*****************************************************************************
 *
 *  dynamic_spmv_coo.cpp
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

/*! \file dynamic_spmv_coo.cpp
 *  \brief Description
 */
#include <examples/include/parser.hpp>
#include <examples/include/timer.hpp>
#include <examples/include/dynamic.hpp>

using namespace morpheus::examples;

void spMv_bench(int argc, char** argv, std::string format, Matrix A)
{
	parser args;
	args.get(argc, argv).print();

	timer total("Dyn_Total"), io("Dyn_IO"), spmv("Dyn_spMv");

	total.start();

	io.start();
	morpheus::io::read_matrix_market_file(A, args.file);
	io.stop();

	Dense_vector y(A.nrows());

	for (int i = 0; i < args.iterations; i++)
	{
		Random_vector r(A.nrows(), i);
		Dense_vector x(r.begin(), r.end());
		spmv.clear().start();
		morpheus::multiply(A, x, y);
		spmv.stop();
		std::cout << "Iteration " << i << ":\t" << spmv;
	}

	total.stop();

	// Stats
	std::cout << args.filename << "\t" << A.nrows()   << "\t" << A.ncols() << "\t" << A.nnz()  << std::endl;
	std::cout << total << io << std::endl;

}

int main(int argc, char* argv[])
{
	Matrix A;

	A = Coo_matrix();
	spMv_bench(argc, argv, "Coo", A);

	A = Csr_matrix();
	spMv_bench(argc, argv, "Csr", A);

	return 0;
}