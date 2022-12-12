/**
 * Morpheus_Copy.hpp
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

#ifndef MORPHEUS_COPY_HPP
#define MORPHEUS_COPY_HPP

#include <impl/Morpheus_Utils.hpp>
#include <impl/Morpheus_Copy_Impl.hpp>
#include <impl/Dynamic/Morpheus_Copy_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup copy Copy
 * \brief Copy Operations on Containers
 * \ingroup data_management
 * \{
 *
 */

/**
 * @brief Performs a deep copy operation between two containers.
 *
 * @tparam SourceType The type of the source container
 * @tparam DestinationType The type of the destination container
 * @param src The container we are copying from
 * @param dst The container we are copying to
 *
 * \note The SourceType and DestinationType must satisfy the \p
 * is_format_compatible or \p is_format_compatible_different_space checks.
 */
template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst) {
  Morpheus::Impl::copy(src, dst);
}

/**
 * @brief Performs a sliced deep copy operation between two containers.
 *
 * @tparam SourceType The type of the source container
 * @tparam DestinationType The type of the destination container
 * @param src The type of the source container
 * @param dst The container we are copying to
 * @param src_begin The begining of the input slice
 * @param src_end The end of the input slice
 * @param dst_begin The begining of the output slice
 * @param dst_end The end of the output slice
 *
 * \note Both containers must be DenseVector containers.
 */
template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type src_begin,
          const typename SourceType::index_type src_end,
          const typename DestinationType::index_type dst_begin,
          const typename DestinationType::index_type dst_end) {
  static_assert(is_dense_vector_format_container_v<SourceType> &&
                    is_dense_vector_format_container_v<DestinationType>,
                "Both src and dst must be vectors.");
  MORPHEUS_ASSERT((src_end - src_begin) == (dst_end - dst_begin),
                  "Source slice range ("
                      << src_begin << ", " << src_end
                      << ") should be equal to the destination slice range ("
                      << dst_begin << ", " << dst_end << ").");

  Morpheus::Impl::copy(src, dst, src_begin, src_end, dst_begin, dst_end);
}

/**
 * @brief Performs a sliced deep copy operation between two containers.
 *
 * @tparam SourceType The type of the source container
 * @tparam DestinationType The type of the destination container
 * @param src The type of the source container
 * @param dst The container we are copying to
 * @param src_begin The begining of the input/output slice
 * @param src_end The end of the input/output slice
 *
 * \note Both containers must be DenseVector containers.
 */
template <typename SourceType, typename DestinationType>
void copy(const SourceType& src, DestinationType& dst,
          const typename SourceType::index_type begin,
          const typename SourceType::index_type end) {
  Morpheus::copy(src, dst, begin, end, begin, end);
}

/**
 * @brief Performs an indirect copy between two containers using a set of key
 * values.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam KeyType The type of the container with the keys
 * @tparam SourceType The type of the source container
 * @tparam DestinationType The type of the destination container
 * @param keys The set of keys we are copying with
 * @param src The type of the source container
 * @param dst The container we are copying to
 *
 * \note The index used to access the key is used as the index where the value
 * will be stored in the dst container: dst[i] = src[keys[i]];
 *
 * \note All containers must be DenseVector containers.
 */
template <typename ExecSpace, typename KeyType, typename SourceType,
          typename DestinationType>
void copy_by_key(const KeyType keys, const SourceType& src,
                 DestinationType& dst) {
  static_assert(is_dense_vector_format_container_v<SourceType> &&
                    is_dense_vector_format_container_v<DestinationType>,
                "Both src and dst must be vectors.");
  Impl::copy_by_key<ExecSpace>(keys, src, dst);
}
/*! \}  // end of copy group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_COPY_HPP