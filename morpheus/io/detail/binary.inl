/*****************************************************************************
 *
 *  binary.inl
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

/*! \file binary.inl
 *  \brief Description
 */

#ifndef MORPHEUS_IO_DETAIL_BINARY_INL
#define MORPHEUS_IO_DETAIL_BINARY_INL

#include <morpheus/matrix_formats/io/binary.hpp>

namespace morpheus
{
	namespace io
	{
		namespace detail
		{
			struct read_binary_file_fn
			{
				read_binary_file_fn(std::string const& filename) : filename_(filename) {}

				using result_type = void;

				template <typename T>
				result_type operator()(T& mtx) const
				{
					morpheus::io::read_binary_file(mtx, filename_);
				}

				std::string filename_;
			};

			struct write_binary_file_fn
			{
				write_binary_file_fn(std::string const& filename) : filename_(filename) {}

				using result_type = void;

				template <typename T>
				result_type operator()(T const& mtx) const
				{
					morpheus::io::write_binary_file(mtx, filename_);
				}

				std::string filename_;
			};
		}   // end namespace detail

		template <typename Types>
		void read_binary_file(matrix<Types>& mtx, const std::string& filename)
		{
			apply_operation(mtx, detail::read_binary_file_fn(filename));
		}

		template <typename Types>
		void write_binary_file(matrix<Types> const& mtx, const std::string& filename)
		{
			apply_operation(mtx, detail::write_binary_file_fn(filename));
		}

	}   // end namespace io
}   // end namespace morpheus

#endif //MORPHEUS_IO_DETAIL_BINARY_INL