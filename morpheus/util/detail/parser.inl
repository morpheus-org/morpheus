/*****************************************************************************
 *
 *  parser.inl
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

/*! \file parser.inl
 *  \brief Description
 */

#ifndef MORPHEUS_UTIL_DETAIL_PARSER_INL
#define MORPHEUS_UTIL_DETAIL_PARSER_INL

#include <iostream>

namespace morpheus
{
	namespace detail
	{

	}   // end namespace detail


	CommandLineParser& CommandLineParser::get(int argc, char **argv)
	{
		check_args(argc);

		program = argv[0];
		fin = argv[1];
		std::string outdir = argv[2];
		iterations = std::stoi(argv[3]);

		filename = fin.substr(fin.find_last_of("/") + 1, fin.size());
		fx = outdir + "/fx.txt";
		fy = outdir + "/fy.txt";

		return *this;
	}

	CommandLineParser& CommandLineParser::print()
	{
		std::cout << filename << "::\tRunning " << program << std::endl;
		std::cout << filename << "::\tInput File:\t" << fin << std::endl;
		std::cout << filename << "::\tInput vector file(fx):\t" << fx << std::endl;
		std::cout << filename << "::\tOutput vector file(fx):\t" << fy << std::endl;
		std::cout << filename << "::\tRunning spMv for " << iterations << " iterations." << std::endl;

		return *this;
	}

	void CommandLineParser::check_args(int argc)
	{
		if(argc != 4)
		{
			std::cerr << "Please specify the filename to be read and number of spmv iterations.";
			exit(-1);
		}
	}
}   // end namespace morpheus


#endif //MORPHEUS_UTIL_DETAIL_PARSER_INL
