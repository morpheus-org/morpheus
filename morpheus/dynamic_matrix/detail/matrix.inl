/*****************************************************************************
 *
 *  matrix.inl
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

/*! \file matrix.inl
 *  \brief Description
 */

#ifndef MORPHEUS_DYNAMIC_MATRIX_DETAIL_MATRIX_INL
#define MORPHEUS_DYNAMIC_MATRIX_DETAIL_MATRIX_INL

#include <morpheus/config.hpp>
#include <morpheus/dynamic_matrix/apply_operation.hpp>

// #include <iostream>

namespace morpheus
{
	namespace detail
	{
		struct any_type_get_num_rows
		{
			using result_type = size_t;
			template <typename T>
			MORPHEUS_INLINE
			result_type operator()(T& v) const { return v.nrows(); }
			// TODO:: const T& v fails for some reason
		};

		struct any_type_get_num_cols
		{
			using result_type = size_t;
			template <typename T>
			MORPHEUS_INLINE
			result_type operator()(T& v) const { return v.ncols(); }
			// TODO:: const T& v fails for some reason
		};

		struct any_type_get_num_nnz
		{
			using result_type = size_t;
			template <typename T>
			MORPHEUS_INLINE
			result_type operator()(T& v) const { return v.nnz(); }
			// TODO:: const T& v fails for some reason
		};

		struct any_type_get_type
		{
			using result_type = std::string;
			template <typename T>
			MORPHEUS_INLINE
			result_type operator()(T& v) const { return v.type(); }
			// TODO:: const T& v fails for some reason
		};
		
	}   // end namespace detail

	template<typename VariantFormats>
	MORPHEUS_INLINE
	typename matrix<VariantFormats>::reference
	matrix<VariantFormats>::operator=(matrix const& mat)
	{
		std::cout << "copy assignment of variant mat\n";
		formats_.swap(const_cast<VariantFormats&>(mat.types()));
		return *this;
	}

	template<typename VariantFormats>
	template <typename Format>
	MORPHEUS_INLINE
	typename matrix<VariantFormats>::reference
	matrix<VariantFormats>::operator=(Format const& mat)
	{
		std::cout << "copy assignment of mat\n";
		formats_.operator=(mat);
		return *this;
	};

	template<typename VariantFormats>
	MORPHEUS_INLINE
	VariantFormats& 
	matrix<VariantFormats>::types()
	{
		return formats_;
	}

	template<typename VariantFormats>
	MORPHEUS_INLINE
	const VariantFormats& 
	matrix<VariantFormats>::types() const
	{
		return formats_;
	}

	template<typename VariantFormats>
	MORPHEUS_INLINE
	size_t 
	matrix<VariantFormats>::nrows()
	{
		return apply_operation(formats_, detail::any_type_get_num_rows());
	};

	template<typename VariantFormats>
	MORPHEUS_INLINE
	size_t 
	matrix<VariantFormats>::ncols()
	{
		return apply_operation(formats_, detail::any_type_get_num_cols());
	};

	template<typename VariantFormats>
	MORPHEUS_INLINE
	size_t 
	matrix<VariantFormats>::nnz()
	{
		return apply_operation(formats_, detail::any_type_get_num_nnz());
	};

	// template<typename VariantFormats>
	// MORPHEUS_INLINE
	// std::string 
	// matrix<VariantFormats>::type()
	// {
	// 	return apply_operation(formats_, detail::any_type_get_type());
	// };

}   // end namespace morpheus

#endif //MORPHEUS_DYNAMIC_MATRIX_DETAIL_MATRIX_INL
