/**
 * Morpheus_Convert_Impl.hpp
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

#include <impl/DenseVector/HIP/Morpheus_Workspace.hpp>

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

namespace Morpheus {
namespace Impl {

HIPWorkspace hipdotspace;

#ifdef MORPHEUS_ENABLE_TPL_HIPBLAS
HIPblasWorkspace hipblasdotspace;
#endif  // MORPHEUS_ENABLE_TPL_HIPBLAS

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
