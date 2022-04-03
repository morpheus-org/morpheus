/**
 * Morpheus_Multiply_Impl.hpp
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

#ifndef MORPHEUS_IMPL_MULTIPLY_IMPL_HPP
#define MORPHEUS_IMPL_MULTIPLY_IMPL_HPP

#include <impl/Coo/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_Multiply_Impl.hpp>
#include <impl/Coo/Kokkos/Morpheus_Multiply_Impl.hpp>

#include <impl/Csr/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Csr/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Csr/Cuda/Morpheus_Multiply_Impl.hpp>
#include <impl/Csr/Kokkos/Morpheus_Multiply_Impl.hpp>

#include <impl/Dia/Serial/Morpheus_Multiply_Impl.hpp>
#include <impl/Dia/OpenMP/Morpheus_Multiply_Impl.hpp>
#include <impl/Dia/Cuda/Morpheus_Multiply_Impl.hpp>
#include <impl/Dia/Kokkos/Morpheus_Multiply_Impl.hpp>

#include <impl/Dynamic/Morpheus_Multiply_Impl.hpp>

#endif  // MORPHEUS_IMPL_MULTIPLY_IMPL_HPP