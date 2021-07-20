/**
 * Morpheus_Print_Impl.hpp
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

#ifndef MORPHEUS_PRINT_IMPL_HPP
#define MORPHEUS_PRINT_IMPL_HPP

#include <iostream>
#include <iomanip>    // setw, setprecision
#include <algorithm>  // max, min
#include <variant>    // visit

#include <impl/Morpheus_FormatTags.hpp>

namespace Morpheus {

// forward decl
template <typename Printable>
void print(const Printable& p);

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s);

namespace Impl {
template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, CooTag) {
  using IndexType = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";

  for (IndexType n = 0; n < p.nnnz(); n++) {
    s << " " << std::setw(14) << p.row_indices[n];
    s << " " << std::setw(14) << p.column_indices[n];
    s << " " << std::setprecision(4) << std::setw(8) << "(" << p.values[n]
      << ")\n";
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, CsrTag) {
  using IndexType = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";

  for (IndexType i = 0; i < p.nrows(); i++) {
    for (IndexType jj = p.row_offsets[i]; jj < p.row_offsets[i + 1]; jj++) {
      s << " " << std::setw(14) << i;
      s << " " << std::setw(14) << p.column_indices[jj];
      s << " " << std::setprecision(4) << std::setw(8) << "(" << p.values[jj]
        << ")\n";
    }
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, DiaTag) {
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";

  using IndexType       = typename Printable::index_type;
  using ValueType       = typename Printable::value_type;
  const IndexType ndiag = p.values.ncols();

  for (IndexType i = 0; i < ndiag; i++) {
    const IndexType k = p.diagonal_offsets[i];

    const IndexType i_start = std::max<IndexType>(0, -k);
    const IndexType j_start = std::max<IndexType>(0, k);

    // number of elements to process in this diagonal
    const IndexType N = std::min(p.nrows() - i_start, p.ncols() - j_start);

    for (IndexType n = 0; n < N; n++) {
      ValueType temp = p.values(i_start + n, i);
      if (temp != ValueType(0)) {
        s << " " << std::setw(14) << i_start + n;
        s << " " << std::setw(14) << i;
        s << " " << std::setprecision(4) << std::setw(8) << "(" << temp
          << ")\n";
      }
    }
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, DenseMatrixTag) {
  using IndexType = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nrows() * p.ncols() << " entries\n";

  for (IndexType i = 0; i < p.nrows(); i++) {
    for (IndexType j = 0; j < p.ncols(); j++) {
      s << " " << std::setw(14) << i;
      s << " " << std::setw(14) << j;
      s << " " << std::setprecision(4) << std::setw(8) << "(" << p(i, j)
        << ")\n";
    }
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, DenseVectorTag) {
  using IndexType = typename Printable::index_type;
  s << p.name() << "<" << p.size() << "> with " << p.size() << " entries\n";

  for (IndexType n = 0; n < p.size(); n++) {
    s << " " << std::setw(14) << n;
    s << " " << std::setprecision(4) << std::setw(8) << "(" << p[n] << ")\n";
  }
}

struct print_fn {
  using result_type = void;
  // TODO: Specify the stream to print to
  // template <typename T, typename S>
  // result_type operator()(const T& mat, S& s) const {
  //   Morpheus::print(mat, s);
  // }
  template <typename Printable>
  result_type operator()(const Printable& p) const {
    Morpheus::print(p);
  }
};

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, DynamicTag) {
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";
  // TODO: Using a stream in this way doesn't seem to work
  // std::visit(std::bind(Impl::print_fn(), std::placeholders::_1,
  // std::ref(s)),
  //            p.formats());
  std::visit(Impl::print_fn(), p.formats());
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_PRINT_IMPL_HPP