/**
 * Morpheus_Utils.hpp
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

#ifndef MORPHEUS_UTILS_HPP
#define MORPHEUS_UTILS_HPP

#include <Morpheus_Macros.hpp>

#include <iostream>

namespace Morpheus {
namespace Impl {

template <typename Printable, typename Stream>
void print_matrix_header(const Printable& p, Stream& s) {
  s << "<" << p.nrows() << ", " << p.ncols() << "> with " << p.nnnz()
    << " entries\n";
}

template <typename Size1, typename Size2>
MORPHEUS_INLINE_FUNCTION Size1 DIVIDE_INTO(Size1 N, Size2 granularity) {
  return (N + (granularity - 1)) / granularity;
}

template <typename T>
MORPHEUS_INLINE_FUNCTION T min(T x, T y) {
  return x < y ? x : y;
}

template <typename T>
MORPHEUS_INLINE_FUNCTION T max(T x, T y) {
  return x > y ? y : x;
}

template <typename IndexType>
MORPHEUS_INLINE_FUNCTION bool isPow2(
    IndexType x,
    typename std::enable_if<std::is_integral<IndexType>::value>::type* =
        nullptr) {
  return ((x & (x - 1)) == 0);
}

template <typename IndexType>
MORPHEUS_INLINE_FUNCTION IndexType
nextPow2(IndexType x,
         typename std::enable_if<std::is_integral<IndexType>::value>::type* =
             nullptr) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#ifndef NDEBUG
#define MORPHEUS_ASSERT(condition, message)                              \
  do {                                                                   \
    if (!(condition)) {                                                  \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                << " line " << __LINE__ << ": " << message << std::endl; \
      std::terminate();                                                  \
    }                                                                    \
  } while (false)
#else
#define MORPHEUS_ASSERT(condition, message) \
  do {                                      \
  } while (false)
#endif

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_UTILS_HPP