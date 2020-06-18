/*****************************************************************************
 *
 *  static_spmv_coo.cpp
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

/*! \file static_spmv_coo.cpp
 *  \brief Description
 */

#include <iostream>
#include <string>
#include <chrono>

#include <morpheus/matrix_formats/io/matrix_market.hpp>
#include <morpheus/matrix_formats/coo_matrix.hpp>
#include <morpheus/matrix_formats/dense_vector.hpp>
#include <morpheus/matrix_formats/multiply.hpp>
#include <morpheus/memory.hpp>

int main(int argc, char* argv[])
{
	// Command line stuff
	if(argc != 3)
	{
		std::cerr << "Please specify the filename to be read and number of spmv iterations.";
		return -1;
	}

	std::string fin(argv[1]); int count = std::stoi(argv[2]);
	std::string filename = fin.substr(fin.find_last_of("/") + 1, fin.size());

	std::cout << argv[0] << "::\tFile:\t" << fin << "\tIterations:\t" << count << std::endl;

	// Timer stuff
	using timer_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
	using duration_t = std::chrono::duration<double>;
	using sample = std::chrono::high_resolution_clock;

	timer_t timer, spmvtimer, iotimer;
	duration_t elapsedDur, spmvDur, ioDur;
	double elapsedTime = 0.0, ioTime = 0.0, spmvTime = 0.0;

	// Morpheus stuff
	using host = morpheus::host_memory;
	using Coo_matrix = morpheus::coo_matrix<int, float, host>;
	using Dense_vector = morpheus::dense_vector<float, host>;

	timer = sample::now();

	Coo_matrix A;

	iotimer = sample::now();

	morpheus::io::read_matrix_market_file(A, fin);

	ioDur = sample::now() - iotimer;
	ioTime += ioDur.count();

	// random vector
	cusp::random_array<float> r(A.nrows(), 0);

	Dense_vector x(r.begin(), r.end()), y(A.nrows());

	for(int i = 0; i < count; i++)
	{
		spmvtimer = sample::now();

		morpheus::multiply(A, x, y);

		spmvDur = sample::now() - spmvtimer;
		spmvTime += spmvDur.count();

		x = y; // Just copy the result back as input
	}

	elapsedDur = sample::now() - timer;
	elapsedTime += elapsedDur.count();

	// Stats
	std::cout << filename << "\t" << A.nrows()   << "\t" << A.ncols() << "\t" << A.nnz()  << std::endl;
	std::cout << "Timing" << "\t" << elapsedTime << "\t" << ioTime    << "\t" << spmvTime << std::endl;

	return 0;
}

