/*****************************************************************************
 *
 *  matrix.hpp
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

/*! \file matrix.hpp
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_MATRIX_HPP
#define MORPHEUS_DYNAMIC_MATRIX_MATRIX_HPP

// #include <string>
namespace morpheus
{

	template<typename VariantFormats>
    class matrix{
        VariantFormats formats_;
    
    public:

		using reference = matrix&;
        using const_reference = const matrix&;

        matrix() = default;
		
		matrix(matrix const& mat) : formats_(mat.types())
		{}

		template <typename Format>
		explicit matrix(Format const& mat) : formats_(mat)
		{}

		reference operator=(matrix const& mat);

        template <typename Format>
		reference operator=(Format const& mat);

        VariantFormats& types();
        const VariantFormats& types() const;

        // void activate(int format);
        size_t nrows();
		size_t ncols();
		size_t nnz();
        // std::string type();
    };

}   // end namespace morpheus

#include <morpheus/dynamic_matrix/detail/matrix.inl>

#endif //MORPHEUS_DYNAMIC_MATRIX_MATRIX_HPP
