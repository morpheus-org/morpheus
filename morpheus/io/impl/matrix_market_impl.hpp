/**
 * matrix_market_impl.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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

#ifndef MORPHEUS_IO_MATRIX_MARKET_IMPL_HPP
#define MORPHEUS_IO_MATRIX_MARKET_IMPL_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <morpheus/core/exceptions.hpp>

namespace Morpheus {
namespace Io {
namespace Impl {

inline void tokenize(std::vector<std::string>& tokens, const std::string& str,
                     const std::string& delimiters = "\n\r\t ") {
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

struct matrix_market_banner {
  std::string storage;   // "array" or "coordinate"
  std::string symmetry;  // "general", "symmetric"
                         // "hermitian", or "skew-symmetric"
  std::string type;      // "complex", "real", "integer", or "pattern"
};

template <typename Stream>
void read_matrix_market_banner(matrix_market_banner& banner, Stream& input) {
  std::string line;
  std::vector<std::string> tokens;

  // read first line
  std::getline(input, line);
  Impl::tokenize(tokens, line);

  if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" ||
      tokens[1] != "matrix")
    throw Morpheus::IOException("invalid MatrixMarket banner");

  banner.storage  = tokens[2];
  banner.type     = tokens[3];
  banner.symmetry = tokens[4];

  if (banner.storage != "array" && banner.storage != "coordinate")
    throw Morpheus::IOException("invalid MatrixMarket storage format [" +
                                banner.storage + "]");

  if (banner.type != "complex" && banner.type != "real" &&
      banner.type != "integer" && banner.type != "pattern")
    throw Morpheus::IOException("invalid MatrixMarket data type [" +
                                banner.type + "]");

  if (banner.symmetry != "general" && banner.symmetry != "symmetric" &&
      banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
    throw Morpheus::IOException("invalid MatrixMarket symmetry [" +
                                banner.symmetry + "]");
}

template <typename Matrix, typename Stream, typename Format>
void read_matrix_market_stream(Matrix& mtx, Stream& input, Format) {
  // general case
  using IndexType = typename Matrix::index_type;
  using ValueType = typename Matrix::value_type;
  using Space     = typename Kokkos::Serial;

  // read banner
  matrix_market_banner banner;
  read_matrix_market_banner(banner, input);

  if (banner.storage == "coordinate") {
    Morpheus::CooMatrix<IndexType, ValueType, Space> temp;
    // TODO:
    read_coordinate_stream(temp, input, banner);
    // TODO:
    Morpheus::convert(temp, mtx);
  } else  // banner.storage == "array"
  {
    Morpheus::DenseMatrix<ValueType, Space> temp;
    // TODO:
    read_array_stream(temp, input, banner);
    // TODO:
    Morpheus::convert(temp, mtx);
  }
}

}  // namespace Impl

template <typename Matrix>
void read_matrix_market(Matrix& mtx, const std::string& filename) {
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

  Morpheus::Io::read_matrix_market(mtx, file_string);
#else
  Morpheus::Io::read_matrix_market(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
void read_matrix_market(Matrix& mtx, Stream& input) {
  Morpheus::Io::Impl::read_matrix_market_stream(mtx, input,
                                                typename Matrix::format());

  throw Morpheus::NotImplementedException{
      "void read(Matrix& mtx, Stream& input)"};
}

// template <typename Matrix>
// void write_matrix_market(const Matrix& mtx, const std::string& filename) {
//   std::ofstream file(filename.c_str());

//   if (!file)
//     throw cusp::io_exception(std::string("unable to open file \"") + filename
//     +
//                              std::string("\" for writing"));

// #ifdef __APPLE__
//   // WAR OSX-specific issue using rdbuf
//   std::stringstream file_string(std::stringstream::in |
//   std::stringstream::out);

//   cusp::io::write_matrix_market_stream(mtx, file_string);

//   file.rdbuf()->sputn(file_string.str().c_str(), file_string.str().size());
// #else
//   cusp::io::write_matrix_market_stream(mtx, file);
// #endif

//   throw Morpheus::NotImplementedException{
//       "void write(const Matrix& mtx, const std::string& filename)"};
// }

// template <typename Matrix, typename Stream>
// void write_matrix_market(const Matrix& mtx, Stream& output) {
//   cusp::io::detail::write_matrix_market_stream(mtx, output,
//                                                typename Matrix::format());

//   throw Morpheus::NotImplementedException{
//       "void write(const Matrix& mtx, Stream& output)"};
// }
}  // namespace Io
}  // namespace Morpheus

#endif  // MORPHEUS_IO_MATRIX_MARKET_IMPL_HPP