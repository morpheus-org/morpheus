/**
 * Morpheus_Core.cpp
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

#include <iostream>

namespace Morpheus {

void initialize(int& argc, char* argv[]) { Kokkos::initialize(argc, argv); }

void print_configuration(std::ostream& out, const bool detail) {
  std::ostringstream msg;

  msg << "Morpheus Version:" << std::endl;
  msg << "  " << Morpheus_VERSION_MAJOR << "." << Morpheus_VERSION_MINOR << "."
      << Morpheus_VERSION_PATCH << " (" << MORPHEUS_VERSION << ")" << std::endl;

  Kokkos::print_configuration(msg, detail);

  if (detail == true) {
    msg << "Default Host Execution Space Configuration:" << std::endl;
    Kokkos::DefaultHostExecutionSpace::print_configuration(msg, detail);
    msg << "Default Execution Space Configuration:" << std::endl;
    Kokkos::DefaultExecutionSpace::print_configuration(msg, detail);
  }

  out << msg.str() << std::endl;
}
void initialize(InitArguments args) { Kokkos::initialize(args); }

void finalize() { Kokkos::finalize(); }

}  // namespace Morpheus