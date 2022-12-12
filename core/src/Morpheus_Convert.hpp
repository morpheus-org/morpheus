/**
 * Morpheus_Convert.hpp
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

#ifndef MORPHEUS_CONVERT_HPP
#define MORPHEUS_CONVERT_HPP

#include <Morpheus_FormatTags.hpp>
#include <fwd/Morpheus_Fwd_CooMatrix.hpp>

#include <impl/Morpheus_Convert_Impl.hpp>
#include <impl/Dynamic/Morpheus_Convert_Impl.hpp>

namespace Morpheus {
/**
 * \addtogroup conversions Conversions
 * \brief Conversion Operations on Containers
 * \ingroup data_management
 * \{
 *
 */

/**
 * @brief Performs a conversion operation between two containers.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam SourceType The type of the source container
 * @tparam DestinationType The type of the destination container
 * @param src The source container we are converting from
 * @param dst The destination container we are converting to
 *
 */
template <typename ExecSpace, typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst) {
  Impl::convert<ExecSpace>(src, dst);
}

/**
 * @brief Performs an in-place conversion operation of the DynamicMatrix
 * container to the format indicated by the enum index.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam SourceType The type of the source container
 * @param src The source container to convert
 * @param index The enum index of the format to convert to
 *
 * \note The src container must be a DynamicMatrix for the conversion to take
 * place.
 *
 * \note Internally the in-place conversion is achieved using a temporary
 * CooMatrix container.
 *
 * \note In case the conversion fails internally it throws
 * an error and the state of the input container is maintained.
 */
template <typename ExecSpace, typename SourceType>
void convert(SourceType& src, const formats_e index) {
  static_assert(Morpheus::is_dynamic_matrix_format_container<SourceType>::value,
                "Container must be a DynamicMatrix.");
  Morpheus::CooMatrix<
      typename SourceType::value_type, typename SourceType::index_type,
      typename SourceType::array_layout, typename SourceType::execution_space>
      temp;

  try {
    Impl::convert<ExecSpace>(src, temp);
  } catch (...) {
    std::cout << "Warning: Conversion failed! Active state set to: "
              << src.active_index() << std::endl;

    return;
  }

  SourceType dynamic_temp;
  dynamic_temp.activate(index);

  try {
    Impl::convert<ExecSpace>(temp, dynamic_temp);
  } catch (...) {
    std::cout << "Warning: Conversion failed! Active state set to: "
              << src.active_index() << std::endl;

    return;
  }

  src = dynamic_temp;
}

/**
 * @brief Performs an in-place conversion operation of the DynamicMatrix
 * container to the format indicated by the index.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam SourceType The type of the source container
 * @param src The source container to convert
 * @param index The index of the container we will be converting to
 *
 */
template <typename ExecSpace, typename SourceType>
void convert(SourceType& src, const int index) {
  Morpheus::convert<ExecSpace>(src, static_cast<formats_e>(index));
}
/*! \}  // end of conversion group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_CONVERT_HPP