/**
 * Morpheus_ArmUtils.hpp
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

#ifndef MORPHEUS_ARM_UTILS_HPP
#define MORPHEUS_ARM_UTILS_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_TPL_ARMPL)

#include <impl/Morpheus_Utils.hpp>

#include <stdio.h>
#include <stdlib.h>

namespace Morpheus {
namespace Impl {
#define CHECK_ARMPL_ERROR(val) check_armpl((val), #val, __FILE__, __LINE__)
template <typename T>
void check_armpl(T err, const char* const func, const char* const file,
                 const int line) {
  if (err != ARMPL_STATUS_SUCCESS) {
    std::cerr << "ArmPL Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << func << " returned error code " << err << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_TPL_ARMPL
#endif  // MORPHEUS_ARM_UTILS_HPP