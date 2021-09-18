/**
 * Benchmarks_Copy.cpp
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

#include <Morpheus_Core.hpp>

// TODO:
// - Load data on Coo.
// - COO: COO_DEEP, COO_ELEM.
// - CSR: CSR_DEEP, CSR_ELEM.
// - DIA: DIA_DEEP, DIA_ELEM.
// - DYNCOO: COO_DEEP, COO_ELEM.
// - DYNCSR: CSR_DEEP, CSR_ELEM.
// - DYNDIA: DIA_DEEP, DIA_ELEM.
// - Bench copy.

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {}
  Morpheus::finalize();

  return 0;
}