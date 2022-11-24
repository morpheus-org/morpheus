/**
 * Morpheus_Core.hpp
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

#ifndef MORPHEUS_CORE_HPP
#define MORPHEUS_CORE_HPP

#include <Morpheus_Macros.hpp>

#include <Kokkos_Core.hpp>

// Containers
#include <Morpheus_CooMatrix.hpp>
#include <Morpheus_CsrMatrix.hpp>
#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DiaMatrix.hpp>
#include <Morpheus_DynamicMatrix.hpp>

// Algorithms
#include <Morpheus_Convert.hpp>
#include <Morpheus_Copy.hpp>
#include <Morpheus_Dot.hpp>
#include <Morpheus_MatrixOperations.hpp>
#include <Morpheus_Multiply.hpp>
#include <Morpheus_Print.hpp>
#include <Morpheus_Reduction.hpp>
#include <Morpheus_Scan.hpp>
#include <Morpheus_Sort.hpp>
#include <Morpheus_WAXPBY.hpp>

// Functionality
#include <Morpheus_Spaces.hpp>
#include <Morpheus_MirrorContainers.hpp>
#include <Morpheus_TypeTraits.hpp>

// I/O
#include <Morpheus_MatrixMarket.hpp>

// Other
#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_Metaprogramming.hpp>
#include <Morpheus_ContainerFactory.hpp>

// // Spaces
// #include <spaces/Morpheus_Serial.hpp>

//! Generic Morpheus interfaces
namespace Morpheus {

struct InitArguments : public Kokkos::InitArguments {
  using base = Kokkos::InitArguments;
  int dynamic_format;

  InitArguments(int dyn_fmt = 0, int nt = -1, int nn = -1, int dv = -1,
                bool dw = false, bool ti = false)
      : base(nt, nn, dv, dw, ti), dynamic_format(dyn_fmt) {}
};

void initialize(int& argc, char* argv[], bool banner = true);
void initialize(int& argc, char* argv[], InitArguments& args,
                bool banner = true);
void print_configuration(std::ostream& out, const bool detail = true);
void initialize(InitArguments args = InitArguments(), bool banner = true);
void finalize();

}  // namespace Morpheus

#endif  // MORPHEUS_CORE_HPP