/**
 * Morpheus_Scan_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_SERIAL_SCAN_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_SERIAL_SCAN_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_SERIAL)

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Spaces.hpp>
#include <Morpheus_FormatTags.hpp>

namespace Morpheus {
namespace Impl {

template <typename ExecSpace, typename Vector>
void inclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start,
                    typename std::enable_if_t<
                        Morpheus::is_dense_vector_format_container_v<Vector> &&
                        Morpheus::is_custom_backend_v<ExecSpace> &&
                        Morpheus::has_serial_execution_space_v<ExecSpace> &&
                        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using IndexType = typename Vector::index_type;

  out[start] = in[start];
  for (IndexType i = start + 1; i < start + size; i++) {
    out[i] = out[i - 1] + in[i];
  }
}

template <typename ExecSpace, typename Vector>
void exclusive_scan(const Vector& in, Vector& out,
                    typename Vector::index_type size,
                    typename Vector::index_type start,
                    typename std::enable_if_t<
                        Morpheus::is_dense_vector_format_container_v<Vector> &&
                        Morpheus::is_custom_backend_v<ExecSpace> &&
                        Morpheus::has_serial_execution_space_v<ExecSpace> &&
                        Morpheus::has_access_v<ExecSpace, Vector>>* = nullptr) {
  using IndexType = typename Vector::index_type;
  using ValueType = typename Vector::value_type;

  if (size > 0) {
    out[start] = ValueType(0);
    for (IndexType i = start + 1; i < start + size; i++) {
      out[i] = out[i - 1] + in[i - 1];
    }
  }
}

template <typename ExecSpace, typename Vector1, typename Vector2>
void inclusive_scan_by_key(
    const Vector1& keys, const Vector2& in, Vector2& out,
    typename Vector2::index_type size, typename Vector2::index_type start,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using IndexType = typename Vector2::index_type;
  using KeyType   = typename Vector1::value_type;
  using ValueType = typename Vector2::value_type;

  KeyType prev_key     = keys[start];
  ValueType prev_value = in[start];
  out[start]           = in[start];

  for (IndexType i = start + 1; i < start + size; i++) {
    KeyType key = keys[i];

    if (prev_key == key)
      out[i] = prev_value = prev_value + in[i];
    else
      out[i] = prev_value = in[i];

    prev_key = key;
  }
}

template <typename ExecSpace, typename Vector1, typename Vector2>
void exclusive_scan_by_key(
    const Vector1& keys, const Vector2& in, Vector2& out,
    typename Vector2::index_type size, typename Vector2::index_type start,
    typename std::enable_if_t<
        Morpheus::is_dense_vector_format_container_v<Vector1> &&
        Morpheus::is_dense_vector_format_container_v<Vector2> &&
        Morpheus::is_custom_backend_v<ExecSpace> &&
        Morpheus::has_serial_execution_space_v<ExecSpace> &&
        Morpheus::has_access_v<ExecSpace, Vector1, Vector2>>* = nullptr) {
  using IndexType = typename Vector2::index_type;
  using KeyType   = typename Vector1::value_type;
  using ValueType = typename Vector2::value_type;

  KeyType temp_key     = keys[start];
  ValueType temp_value = in[start];
  ValueType next       = ValueType(0);

  // first one is init
  out[start] = next;
  next       = next + temp_value;

  for (IndexType i = start + 1; i < start + size; i++) {
    KeyType key = keys[i];

    // use temp to permit in-place scans
    temp_value = in[i];

    if (temp_key != key) next = ValueType(0);  // reset sum

    out[i] = next;
    next   = next + temp_value;

    temp_key = key;
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_SERIAL
#endif  // MORPHEUS_DENSEVECTOR_SERIAL_SCAN_IMPL_HPP