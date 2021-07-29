/**
 * Morpheus_Core.hpp
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

#ifndef MORPHEUS_CORE_HPP
#define MORPHEUS_CORE_HPP

#include <Morpheus_Macros.hpp>

#include <Kokkos_Core.hpp>

#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_CooMatrix.hpp>
#include <Morpheus_CsrMatrix.hpp>
#include <Morpheus_DiaMatrix.hpp>
#include <Morpheus_DynamicMatrix.hpp>

#include <Morpheus_Multiply.hpp>
#include <Morpheus_Print.hpp>
#include <Morpheus_Sort.hpp>

#include <Morpheus_MatrixMarket.hpp>

namespace Morpheus {

struct InitArguments : public Kokkos::InitArguments {
  InitArguments(int nt = -1, int nn = -1, int dv = -1, bool dw = false,
                bool ti = false)
      : Kokkos::InitArguments(nt, nn, dv, dw, ti) {}
};

void initialize(int& argc, char* argv[]);
void print_configuration(std::ostream& out, const bool detail = true);
void initialize(InitArguments args = InitArguments());
void finalize();

}  // namespace Morpheus

#endif  // MORPHEUS_CORE_HPP