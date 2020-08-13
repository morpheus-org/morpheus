/*****************************************************************************
 *
 *  format_selector.hpp
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

/*! \file format_selector.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_FORMAT_SELECTOR_HPP
#define MORPHEUS_FORMAT_SELECTOR_HPP

#include <morpheus/dynamic_matrix/matrix.hpp>

#include <morpheus/memory.hpp>
#include <morpheus/matrix_formats/coo_matrix.hpp>
#include <morpheus/matrix_formats/csr_matrix.hpp>
// #include <morpheus/matrix_formats/dia_matrix.hpp>
#include <morpheus/matrix_formats/ell_matrix.hpp>
// #include <morpheus/matrix_formats/hyb_matrix.hpp>
// #include <morpheus/matrix_formats/dense_matrix.hpp>

namespace morpheus
{
	enum FormatPool { FMT_COO = 0, FMT_CSR, FMT_DIA, FMT_ELL, FMT_HYB, FMT_DENSE };

	template <typename VariantFormats>
	void select_format(const int format, matrix<VariantFormats>& mat)
	{

		switch(format) {
			case FMT_COO :
				mat = morpheus::coo_matrix<int, double, morpheus::host_memory>();
				break;
			case FMT_CSR :
				mat = morpheus::csr_matrix<int, double, morpheus::host_memory>();
				break;
			// case FMT_DIA :
			// 	mat = morpheus::dia_matrix<int, double, morpheus::host_memory>();
			// 	break;
			case FMT_ELL :
				mat = morpheus::ell_matrix<int, double, morpheus::host_memory>();
				break;
			// case FMT_HYB :
			// 	mat = morpheus::hyb_matrix<int, double, morpheus::host_memory>();
			// 	break;
			// case FMT_DENSE :
			// 	mat = morpheus::dense_matrix<double, morpheus::host_memory>();
				break;
			default :
				mat = morpheus::coo_matrix<int, double, morpheus::host_memory>();
		}

	}

}   // end namespace morpheus

#endif //MORPHEUS_FORMAT_SELECTOR_HPP
