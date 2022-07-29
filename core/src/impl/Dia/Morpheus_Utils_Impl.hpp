/**
 * Morpheus_Utils_Impl.hpp
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

#ifndef MORPHEUS_IMPL_DIA_UTILS_IMPL_HPP
#define MORPHEUS_IMPL_DIA_UTILS_IMPL_HPP

#include <Morpheus_Macros.hpp>

namespace Morpheus {
namespace Impl {

// Calculates padding to align the data based on the current length
template <typename T>
MORPHEUS_INLINE_FUNCTION const T get_pad_size(T len, T alignment) {
  return alignment * ((len + alignment - 1) / alignment);
}

/**
 * @brief Checks if the current matrix exceeds a tolerance level reflecting
 * the performance of the DIA format.
 *
 * @param num_rows Number of
 * @param num_entries
 * @param num_diagonals
 * @return bool
 */
template <typename T>
bool exceeds_tolerance(const T num_rows, const T num_entries,
                       const T num_diagonals) {
  const float max_fill   = 10.0;
  const float threshold  = 10e9;  // 100M entries
  const float size       = float(num_diagonals) * float(num_rows);
  const float fill_ratio = size / std::max(1.0f, float(num_entries));

  bool res = false;
  if (max_fill < fill_ratio && size > threshold) {
    res = true;
  }

  return res;
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_IMPL_DIA_UTILS_IMPL_HPP