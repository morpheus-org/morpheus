/**
 * Morpheus_MatrixMarket.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
 *
 * Contributing Authors:
 * Christodoulos Stylianou (c.stylianou@ed.ac.uk)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MORPHEUS_MATRIXMARKET_HPP
#define MORPHEUS_MATRIXMARKET_HPP

#include <impl/Morpheus_MatrixMarket_Impl.hpp>

namespace Morpheus {
namespace IO {

template <typename Container, typename Stream>
void read_matrix_market_stream(Container& container, Stream& input) {
  static_assert(
      Morpheus::is_coo_matrix_format_container_v<Container> ||
          Morpheus::is_dynamic_matrix_format_container_v<Container> ||
          Morpheus::is_dense_matrix_format_container_v<Container> ||
          Morpheus::is_dense_vector_format_container_v<Container>,
      "Container must be either a CooMatrix, DynamicMatrix, DenseMatrix "
      "or DenseVector container");
  static_assert(Morpheus::has_host_memory_space_v<Container>,
                "Container must reside in HostSpace");
  Morpheus::IO::Impl::read_matrix_market_stream(container, input);
}

template <typename Container>
void read_matrix_market_file(Container& container,
                             const std::string& filename) {
  static_assert(
      Morpheus::is_coo_matrix_format_container_v<Container> ||
          Morpheus::is_dynamic_matrix_format_container_v<Container> ||
          Morpheus::is_dense_matrix_format_container_v<Container> ||
          Morpheus::is_dense_vector_format_container_v<Container>,
      "Container must be either a CooMatrix, DynamicMatrix, DenseMatrix "
      "or DenseVector container");
  static_assert(Morpheus::has_host_memory_space_v<Container>,
                "Container must reside in HostSpace");
  std::ifstream file(filename.c_str());

  if (!file)
    throw Morpheus::IOException(std::string("unable to open file \"") +
                                filename + std::string("\" for reading"));

#ifdef __APPLE__
  // WAR OSX-specific issue using rdbuf
  std::stringstream file_string(std::stringstream::in | std::stringstream::out);
  std::vector<char> buffer(
      file.rdbuf()->pubseekoff(0, std::ios::end, std::ios::in));
  file.rdbuf()->pubseekpos(0, std::ios::in);
  file.rdbuf()->sgetn(&buffer[0], buffer.size());
  file_string.write(&buffer[0], buffer.size());

  Morpheus::IO::read_matrix_market_stream(container, file_string);
#else
  Morpheus::IO::read_matrix_market_stream(container, file);
#endif
}

template <typename Container, typename Stream>
void write_matrix_market_stream(const Container& container, Stream& output) {
  static_assert(
      Morpheus::is_coo_matrix_format_container_v<Container> ||
          Morpheus::is_dynamic_matrix_format_container_v<Container> ||
          Morpheus::is_dense_matrix_format_container_v<Container> ||
          Morpheus::is_dense_vector_format_container_v<Container>,
      "Container must be either a CooMatrix, DynamicMatrix, DenseMatrix "
      "or DenseVector container");
  static_assert(Morpheus::has_host_memory_space_v<Container>,
                "Container must reside in HostSpace");
  Morpheus::IO::Impl::write_matrix_market_stream(container, output);
}

template <typename Container>
void write_matrix_market_file(const Container& container,
                              const std::string& filename) {
  static_assert(
      Morpheus::is_coo_matrix_format_container_v<Container> ||
          Morpheus::is_dynamic_matrix_format_container_v<Container> ||
          Morpheus::is_dense_matrix_format_container_v<Container> ||
          Morpheus::is_dense_vector_format_container_v<Container>,
      "Container must be either a CooMatrix, DynamicMatrix, DenseMatrix "
      "or DenseVector container");
  static_assert(Morpheus::has_host_memory_space_v<Container>,
                "Container must reside in HostSpace");
  std::ofstream file(filename.c_str());

  if (!file)
    throw Morpheus::IOException(std::string("unable to open file \"") +
                                filename + std::string("\" for writing"));

#ifdef __APPLE__
  // WAR OSX-specific issue using rdbuf
  std::stringstream file_string(std::stringstream::in | std::stringstream::out);

  Morpheus::IO::write_matrix_market_stream(container, file_string);

  file.rdbuf()->sputn(file_string.str().c_str(), file_string.str().size());
#else
  Morpheus::IO::write_matrix_market_stream(container, file);
#endif
}
}  // namespace IO
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXMARKET_HPP