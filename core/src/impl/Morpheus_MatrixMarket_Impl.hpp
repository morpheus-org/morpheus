/**
 * Morpheus_MatrixMarket_Impl.hpp
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

#ifndef MORPHEUS_MATRIX_MARKET_IMPL_HPP
#define MORPHEUS_MATRIX_MARKET_IMPL_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <Morpheus_Core.hpp>
#include <Morpheus_Exceptions.hpp>
#include <Morpheus_CooMatrix.hpp>
#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_Sort.hpp>
#include <Morpheus_Copy.hpp>

namespace Morpheus {
namespace Io {
// forward decl
template <typename Matrix, typename Stream>
void read_matrix_market_stream(Matrix& mtx, Stream& input);

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

template <typename ValueType, typename... Properties, typename Stream>
void read_coordinate_stream(Morpheus::CooMatrix<ValueType, Properties...>& coo,
                            Stream& input, const matrix_market_banner& banner) {
  using IndexType =
      typename Morpheus::CooMatrix<ValueType, Properties...>::index_type;
  // read file contents line by line
  std::string line;

  // skip over banner and comments
  do {
    std::getline(input, line);
  } while (line[0] == '%');

  // line contains [num_rows num_columns num_entries]
  std::vector<std::string> tokens;
  Impl::tokenize(tokens, line);

  if (tokens.size() != 3)
    throw Morpheus::IOException("invalid MatrixMarket coordinate format");

  IndexType num_rows, num_cols, num_entries;

  std::istringstream(tokens[0]) >> num_rows;
  std::istringstream(tokens[1]) >> num_cols;
  std::istringstream(tokens[2]) >> num_entries;

  coo.resize(num_rows, num_cols, num_entries);

  IndexType num_entries_read = 0;

  // read file contents
  if (banner.type == "pattern") {
    while (num_entries_read < coo.nnnz() && !input.eof()) {
      input >> coo.row_indices[num_entries_read];
      input >> coo.column_indices[num_entries_read];
      num_entries_read++;
    }

    auto values_begin = coo.values.data();
    auto values_end   = coo.values.data() + coo.values.size();
    std::fill(values_begin, values_end, ValueType(1));
  } else if (banner.type == "real" || banner.type == "integer") {
    while (num_entries_read < coo.nnnz() && !input.eof()) {
      ValueType real;

      input >> coo.row_indices[num_entries_read];
      input >> coo.column_indices[num_entries_read];
      input >> real;

      coo.values[num_entries_read] = real;
      num_entries_read++;
    }
  } else if (banner.type == "complex") {
    throw Morpheus::NotImplementedException(
        "Morpheus does not currently support complex "
        "matrices");

  } else {
    throw Morpheus::IOException("invalid MatrixMarket data type");
  }

  if (num_entries_read != coo.nnnz())
    throw Morpheus::IOException(
        "unexpected EOF while reading MatrixMarket entries");

  // check validity of row and column index data
  if (coo.nnnz() > 0) {
    auto row_indices_begin = coo.row_indices.data();
    auto row_indices_end   = coo.row_indices.data() + coo.row_indices.size();
    IndexType min_row_index =
        *std::min_element(row_indices_begin, row_indices_end);
    IndexType max_row_index =
        *std::max_element(row_indices_begin, row_indices_end);

    auto column_indices_begin = coo.column_indices.data();
    auto column_indices_end =
        coo.column_indices.data() + coo.column_indices.size();
    IndexType min_col_index =
        *std::min_element(column_indices_begin, column_indices_end);
    IndexType max_col_index =
        *std::max_element(column_indices_begin, column_indices_end);

    if (min_row_index < 1)
      throw Morpheus::IOException("found invalid row index (index < 1)");
    if (min_col_index < 1)
      throw Morpheus::IOException("found invalid column index (index < 1)");
    if (max_row_index > coo.nrows())
      throw Morpheus::IOException("found invalid row index (index > num_rows)");
    if (max_col_index > coo.ncols())
      throw Morpheus::IOException(
          "found invalid column index (index > num_columns)");
  }

  // convert base-1 indices to base-0
  for (IndexType n = 0; n < coo.nnnz(); n++) {
    coo.row_indices[n] -= 1;
    coo.column_indices[n] -= 1;
  }

  // expand symmetric formats to "general" format
  if (banner.symmetry != "general") {
    IndexType off_diagonals = 0;

    for (IndexType n = 0; n < coo.nnnz(); n++)
      if (coo.row_indices[n] != coo.column_indices[n]) off_diagonals++;

    IndexType general_num_entries = coo.nnnz() + off_diagonals;

    Morpheus::CooMatrix<ValueType, Properties...> general(num_rows, num_cols,
                                                          general_num_entries);

    if (banner.symmetry == "symmetric") {
      IndexType nnz = 0;

      for (IndexType n = 0; n < coo.nnnz(); n++) {
        // copy entry over
        general.row_indices[nnz]    = coo.row_indices[n];
        general.column_indices[nnz] = coo.column_indices[n];
        general.values[nnz]         = coo.values[n];
        nnz++;

        // duplicate off-diagonals
        if (coo.row_indices[n] != coo.column_indices[n]) {
          general.row_indices[nnz]    = coo.column_indices[n];
          general.column_indices[nnz] = coo.row_indices[n];
          general.values[nnz]         = coo.values[n];
          nnz++;
        }
      }
    } else if (banner.symmetry == "hermitian") {
      throw Morpheus::NotImplementedException(
          "MatrixMarket I/O does not currently support hermitian matrices");
      // TODO
    } else if (banner.symmetry == "skew-symmetric") {
      // TODO
      throw Morpheus::NotImplementedException(
          "MatrixMarket I/O does not currently support skew-symmetric "
          "matrices");
    }

    // store full matrix in coo
    // ! FIXME: might be better to do swap
    coo = general;
  }  // if (banner.symmetry != "general")
  // sort indices by (row,column)
  Morpheus::sort_by_row_and_column(coo);
}

template <typename ValueType, typename... Properties, typename Stream>
void read_array_stream(Morpheus::DenseMatrix<ValueType, Properties...>& mtx,
                       Stream& input, const matrix_market_banner& banner) {
  using IndexType =
      typename Morpheus::DenseMatrix<ValueType, Properties...>::index_type;
  // read file contents line by line
  std::string line;

  // skip over banner and comments
  do {
    std::getline(input, line);
  } while (line[0] == '%');

  std::vector<std::string> tokens;
  Impl::tokenize(tokens, line);

  if (tokens.size() != 2)
    throw Morpheus::IOException("invalid MatrixMarket array format");

  IndexType num_rows, num_cols;

  std::istringstream(tokens[0]) >> num_rows;
  std::istringstream(tokens[1]) >> num_cols;

  Morpheus::DenseMatrix<ValueType, Properties...> dense(num_rows, num_cols);

  size_t num_entries      = num_rows * num_cols;
  size_t num_entries_read = 0;
  IndexType nrows = 0, ncols = 0;

  // read file contents
  if (banner.type == "pattern") {
    throw Morpheus::NotImplementedException(
        "pattern array MatrixMarket format is not supported");
  } else if (banner.type == "real" || banner.type == "integer") {
    while (num_entries_read < num_entries && !input.eof()) {
      double real;

      input >> real;

      dense(nrows, ncols) = ValueType(real);

      num_entries_read++;
      ncols++;
      if (num_entries_read % ncols == 0) {
        nrows++;
        ncols = 0;
      }
    }
  } else if (banner.type == "complex") {
    throw Morpheus::NotImplementedException("complex type is not supported");
  } else {
    throw Morpheus::IOException("invalid MatrixMarket data type");
  }

  if (num_entries_read != num_entries)
    throw Morpheus::IOException(
        "unexpected EOF while reading MatrixMarket entries");

  if (banner.symmetry != "general")
    throw Morpheus::NotImplementedException(
        "only general array symmetric MatrixMarket format is supported");

  Morpheus::copy(dense, mtx);
}

template <template <class, class...> class Container, class T, class... P,
          typename Stream>
void read_matrix_market_stream(
    Container<T, P...>& mtx, Stream& input, Morpheus::Impl::SparseMatTag,
    typename std::enable_if<
        is_container<typename Container<T, P...>::tag>::value &&
        is_HostSpace_v<typename Container<T, P...>::memory_space>>::type* =
        nullptr) {
  // read banner
  matrix_market_banner banner;
  read_matrix_market_banner(banner, input);

  if (banner.storage == "coordinate") {
    Morpheus::CooMatrix<T, P...> temp;
    read_coordinate_stream(temp, input, banner);
    Morpheus::convert(temp, mtx);
  } else  // banner.storage == "array"
  {
    Morpheus::DenseMatrix<T, P...> temp;
    read_array_stream(temp, input, banner);
    Morpheus::convert(temp, mtx);
  }
}

template <typename Matrix, typename Stream>
void read_matrix_market_stream(Matrix& mtx, Stream& input,
                               Morpheus::DenseVectorTag) {
  // array1d case
  using ValueType = typename Matrix::value_type;
  using Space     = Kokkos::Serial;

  Morpheus::DenseMatrix<ValueType, Space> temp;

  Morpheus::Io::read_matrix_market_stream(temp, input);

  Morpheus::copy(temp, mtx);
}

template <typename Stream, typename ScalarType>
void write_value(Stream& output, const ScalarType& value) {
  output << value;
}

template <typename ValueType, typename... Properties, typename Stream>
void write_coordinate_stream(
    const Morpheus::CooMatrix<ValueType, Properties...>& coo, Stream& output) {
  if (std::is_floating_point_v<ValueType> || std::is_integral_v<ValueType>) {
    output << "%%MatrixMarket matrix coordinate real general\n";
  } else {
    // output << "%%MatrixMarket matrix coordinate complex general\n";
    throw Morpheus::NotImplementedException("complex type is not supported.");
  }

  output << "\t" << coo.nrows() << "\t" << coo.ncols() << "\t" << coo.nnnz()
         << "\n";

  for (size_t i = 0; i < size_t(coo.nnnz()); i++) {
    output << (coo.row_indices[i] + 1) << " ";
    output << (coo.column_indices[i] + 1) << " ";
    Morpheus::Io::Impl::write_value(output, coo.values[i]);
    output << "\n";
  }
}

template <template <class, class...> class Container, class T, class... P,
          typename Stream>
void write_matrix_market_stream(
    const Container<T, P...>& mtx, Stream& output, Morpheus::Impl::SparseMatTag,
    typename std::enable_if<
        is_container<typename Container<T, P...>::tag>::value &&
        is_Host_Memoryspace_v<typename Container<T, P...>::memory_space>>::
        type* = nullptr) {
  // general sparse case
  Morpheus::CooMatrix<T, P...> coo;
  Morpheus::convert(mtx, coo);

  Morpheus::Io::Impl::write_coordinate_stream(coo, output);
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output,
                                Morpheus::DenseVectorTag) {
  using ValueType = typename Matrix::value_type;

  if (std::is_floating_point_v<ValueType> || std::is_integral_v<ValueType>) {
    output << "%%MatrixMarket matrix array real general\n";
  } else {
    // output << "%%MatrixMarket matrix array complex general\n";
    throw Morpheus::NotImplementedException("complex type is not supported.");
  }

  output << "\t" << mtx.size() << "\t1\n";

  for (size_t i = 0; i < mtx.size(); i++) {
    write_value(output, mtx[i]);
    output << "\n";
  }
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output,
                                Morpheus::DenseMatrixTag) {
  using ValueType = typename Matrix::value_type;

  if (std::is_floating_point_v<ValueType> || std::is_integral_v<ValueType>) {
    output << "%%MatrixMarket matrix array real general\n";
  } else {
    // output << "%%MatrixMarket matrix array complex general\n";
    throw Morpheus::NotImplementedException("complex type is not supported.");
  }

  output << "\t" << mtx.nrows() << "\t" << mtx.ncols() << "\n";

  for (size_t j = 0; j < mtx.nrows(); j++) {
    for (size_t i = 0; i < mtx.ncols(); i++) {
      write_value(output, mtx(i, j));
      output << "\n";
    }
  }
}

}  // namespace Impl
}  // namespace Io
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIX_MARKET_IMPL_HPP