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

#ifndef MORPHEUS_DYNAMIC_MATRIX_HPP
#define MORPHEUS_DYNAMIC_MATRIX_HPP

#include <boost/variant.hpp>
#include <morpheus/dynamic/apply_operation.hpp>

#include <iostream>

namespace morpheus
{
	// TODO:: Allow change of type from a function using an index

	struct any_type_get_num_rows
	{
		using result_type = size_t;
		template <typename T>
		result_type operator()(T& v) const { return v.nrows(); }
		// TODO:: const T& v fails for some reason
	};

	struct any_type_get_num_cols
	{
		using result_type = size_t;
		template <typename T>
		result_type operator()(T& v) const { return v.ncols(); }
		// TODO:: const T& v fails for some reason
	};

	struct any_type_get_num_nnz
	{
		using result_type = size_t;
		template <typename T>
		result_type operator()(T& v) const { return v.nnz(); }
		// TODO:: const T& v fails for some reason
	};

	template<typename Matrices>
	class matrix : public boost::make_variant_over<Matrices>::type
	{
		using parent_t = typename boost::make_variant_over<Matrices>::type;

	public:
		matrix() = default;
		matrix(matrix const& mat) : parent_t((parent_t const&) mat) {std::cout << "matrix(matrix const& mat)" << std::endl;};

		template <typename Matrix>
		explicit matrix(Matrix const& mat) : parent_t(mat) {std::cout << "explicit matrix(Matrix const& mat)" << std::endl;};

		template <typename OtherMatrices>
		matrix(matrix<OtherMatrices> const& mat)
				: parent_t((typename boost::make_variant_over<OtherMatrices>::type const&)mat)
		{}

		matrix& operator=(matrix const& mat)
		{
			std::cout << "matrix& operator=(matrix const& mat)" << std::endl;
			parent_t::operator=((parent_t const&)mat);
			return *this;
		}

		template <typename Matrix>
		matrix& operator=(Matrix const& mat)
		{
			std::cout << "matrix& operator=(Matrix const& mat)" << std::endl;
			parent_t::operator=(mat);
			return *this;
		}

		template <typename OtherMatrices>
		matrix& operator=(matrix<OtherMatrices> const& mat)
		{
			parent_t::operator=((typename boost::make_variant_over<OtherMatrices>::type const&)mat);
			return *this;
		}

		size_t nrows()
		{
			return apply_operation(*this, any_type_get_num_rows());
		}

		size_t ncols()
		{
			return apply_operation(*this, any_type_get_num_cols());
		}

		size_t nnz()
		{
			return apply_operation(*this, any_type_get_num_nnz());
		}
	};
}


#endif //MORPHEUS_DYNAMIC_MATRIX_HPP
