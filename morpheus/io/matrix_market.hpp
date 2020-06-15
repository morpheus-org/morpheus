/*****************************************************************************
 *
 *  matrix_market.hpp
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

/*! \file matrix_market.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_IO_MATRIX_MARKET_HPP
#define MORPHEUS_IO_MATRIX_MARKET_HPP

#include <cusp/io/matrix_market.h>

#include <string>

namespace morpheus
{
	namespace io
	{
		template <typename Matrix>
		void read_matrix_market_file(Matrix& mtx, const std::string& filename)
		{
			cusp::io::read_matrix_market_file(mtx, filename);
		}

		template <typename Matrix>
		void write_matrix_market_file(const Matrix& mtx, const std::string& filename)
		{
			cusp::io::write_matrix_market_file(mtx, filename);
		}
	}
}

#endif //MORPHEUS_IO_MATRIX_MARKET_HPP
