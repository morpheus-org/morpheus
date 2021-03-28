/**
 * print.inl
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

#ifndef MORPHEUS_ALGORITHMS_IMPL_PRINT_INL
#define MORPHEUS_ALGORITHMS_IMPL_PRINT_INL

#include <iostream>
#include <iomanip>

#include <morpheus/core/exceptions.hpp>
#include <morpheus/containers/dynamic_matrix.hpp>
#include <morpheus/containers/vector.hpp>

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, Morpheus::CooTag) {
  using index_type = typename Printable::index_type;
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";

  for (index_type n = 0; n < p.nnnz(); n++) {
    s << " " << std::setw(14) << p.row_indices[n];
    s << " " << std::setw(14) << p.column_indices[n];
    s << " " << std::setprecision(4) << std::setw(8) << "(" << p.values[n]
      << ")\n";
  }
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, Morpheus::Impl::SparseMatTag) {
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";
  Morpheus::NotImplementedException(
      "void print(const Printable& p, Stream& s, "
      "Morpheus::Impl::SparseMatTag)");
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s, Morpheus::DenseVectorTag) {
  using index_type = typename Printable::size_type;
  s << p.name() << "<" << p.size() << "> with " << p.size() << " entries\n";

  for (index_type n = 0; n < p.size(); n++) {
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
void print(const Printable& p, Stream& s, Morpheus::DynamicTag) {
  s << p.name() << "<" << p.nrows() << ", " << p.ncols() << "> with "
    << p.nnnz() << " entries\n";
  // TODO: Using a stream in this way doesn't seem to work
  // std::visit(std::bind(Impl::print_fn(), std::placeholders::_1, s),
  //            p.formats());
  std::visit(Impl::print_fn(), p.formats());
}

}  // namespace Impl

template <typename Printable>
void print(const Printable& p) {
  Morpheus::print(p, std::cout);
}

template <typename Printable, typename Stream>
void print(const Printable& p, Stream& s) {
  Morpheus::Impl::print(p, s, typename Printable::tag());
}

}  // namespace Morpheus

#endif  // MORPHEUS_ALGORITHMS_IMPL_PRINT_INL