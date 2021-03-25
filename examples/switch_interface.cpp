/**
 * switch_interface.cpp
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

#include <iostream>
#include <morpheus/containers/dynamic_matrix.hpp>

int main() {
  Morpheus::DynamicMatrix<double, int> A;
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // Switch through available enums
  A.activate(Morpheus::CSR_FORMAT);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  A.activate(Morpheus::DIA_FORMAT);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // Switch using integer indexing
  A.activate(0);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  // In case indexing exceeds the size of available types in the underlying
  // variant, we default to the first entry
  A.activate(5);
  std::cout << "Active Index: " << A.active_index() << std::endl;

  return 0;
}