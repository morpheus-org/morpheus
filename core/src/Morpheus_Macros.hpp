/**
 * Morpheus_Macros.hpp
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

#ifndef MORPHEUS_MACROS_HPP
#define MORPHEUS_MACROS_HPP

#include <MorpheusCore_config.hpp>

#if defined(MORPHEUS_ENABLE_CUDA) || defined(MORPHEUS_ENABLE_HIP)
#define MORPHEUS_INLINE_FUNCTION inline __device__ __host__
#define MORPHEUS_LAMBDA [=] __device__
#else
#define MORPHEUS_INLINE_FUNCTION inline
#define MORPHEUS_LAMBDA [=]
#endif

#define MORPHEUS_FORCEINLINE_FUNCTION KOKKOS_FORCEINLINE_FUNCTION

#endif  // MORPHEUS_MACROS_HPP