/**
 * Morpheus_Convert.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#include <Morpheus_Print.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_ContainerFactory.hpp>
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
 * @brief Enum values for in-place conversion status
 *
 */
enum conversion_error_e {
  DYNAMIC_TO_PROXY = 0,  //!< Dynamic to Proxy conversion failed
  PROXY_TO_DYNAMIC,      //!< Proxy to Dynamic conversion failed
  CONV_SUCCESS           //!< In-place conversion successful
};

/**
 * @brief Performs an in-place conversion operation of the DynamicMatrix
 * container to the format indicated by the enum index.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam SourceType The type of the source container
 * @param src The source container to convert
 * @param index The enum index of the format to convert to
 * @return conversion_error_e Error code
 *
 * \note The src container must be a DynamicMatrix for the conversion to take
 * place.
 *
 * \note Internally the in-place conversion is achieved using a temporary
 * CooMatrix container.
 *
 * \note In case the conversion fails internally it throws
 * an error and the state and data of the input container are maintained.
 */
template <typename ExecSpace, typename SourceType>
conversion_error_e convert(SourceType& src, const formats_e index) {
  static_assert(Morpheus::is_dynamic_matrix_format_container<SourceType>::value,
                "Container must be a DynamicMatrix.");
  using value_type = typename SourceType::value_type;
  using CooMatrix =
      typename Morpheus::mirror_params<SourceType,
                                       Morpheus::CooMatrix<value_type>>::type;

  CooMatrix temp;

  // No conversion needed
  if (src.active_index() == index) {
    return CONV_SUCCESS;
  }

  if (src.active_index() != Morpheus::COO_FORMAT) {
    try {
      Impl::convert<ExecSpace>(src, temp);
    } catch (...) {
      return DYNAMIC_TO_PROXY;
    }
  } else {
    temp = src;
  }

  // TODO: If index==COO, shallow copy instead of convert
  SourceType dynamic_temp;
  dynamic_temp.activate(index);

  try {
    Impl::convert<ExecSpace>(temp, dynamic_temp);
  } catch (...) {
    return PROXY_TO_DYNAMIC;
  }
  src = dynamic_temp;
  return CONV_SUCCESS;
}

/**
 * @brief Performs an in-place conversion operation of the DynamicMatrix
 * container to the format indicated by the index.
 *
 * @tparam ExecSpace Execution space to run the algorithm
 * @tparam SourceType The type of the source container
 * @param src The source container to convert
 * @param index The index of the container we will be converting to
 * @return conversion_error_e Error code
 *
 */
template <typename ExecSpace, typename SourceType>
conversion_error_e convert(SourceType& src, const int index) {
  return Morpheus::convert<ExecSpace>(src, static_cast<formats_e>(index));
}
/*! \}  // end of conversion group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_CONVERT_HPP