/**
 * macros.hpp
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

#ifndef MORPHEUS_CORE_MACROS_HPP
#define MORPHEUS_CORE_MACROS_HPP

#ifdef KOKKOS_ENABLE_CUDA
#define MORPHEUS_ENABLE_CUDA  // Kokkos::Cuda execution and memory spaces
#endif

#ifdef KOKKOS_ENABLE_OPENMP
#define MORPHEUS_ENABLE_OPENMP  // Kokkos::OpenMP execution and memory spaces
#endif

#endif  // MORPHEUS_CORE_MACROS_HPP