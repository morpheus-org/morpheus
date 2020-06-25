/*****************************************************************************
 *
 *  parser.hpp
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

/*! \file parser.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_UTIL_PARSER_HPP
#define MORPHEUS_UTIL_PARSER_HPP

#include <string>

namespace morpheus
{
	class CommandLineParser
	{
	private:
		void check_args(int argc);

	public:
		std::string program;
		std::string fin, filename;
		std::string fx, fy;
		int iterations;

		CommandLineParser() = default;

		CommandLineParser& get(int argc, char **argv);
		CommandLineParser& print();
	};
}

#include <morpheus/util/detail/parser.inl>

#endif //MORPHEUS_UTIL_PARSER_HPP
