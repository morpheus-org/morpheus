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

#include <string>

#include <morpheus/core/exceptions.hpp>

namespace Morpheus {
namespace Io {
namespace Impl {}
template <typename Matrix>
void read(Matrix& mtx, const std::string& filename) {
  throw Morpheus::NotImplementedException{
      "void read(Matrix& mtx, const std::string& filename)"};
}
template <typename Matrix, typename Stream>
void read(Matrix& mtx, Stream& input) {
  throw Morpheus::NotImplementedException{
      "void read(Matrix& mtx, Stream& input)"};
}

template <typename Matrix>
void write(const Matrix& mtx, const std::string& filename) {
  throw Morpheus::NotImplementedException{
      "void write(const Matrix& mtx, const std::string& filename)"};
}

template <typename Matrix, typename Stream>
void write(const Matrix& mtx, Stream& output) {
  throw Morpheus::NotImplementedException{
      "void write(const Matrix& mtx, Stream& output)"};
}
}  // namespace Io
}  // namespace Morpheus

#endif  // MORPHEUS_IO_MATRIX_MARKET_IMPL_HPP