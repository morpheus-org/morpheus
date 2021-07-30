/**
 * Examples_DynamicResize.cpp
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

template <typename... Properties>
void stats(Morpheus::DynamicMatrix<Properties...>& mat, std::string name,
           std::string fn_name) {
  std::cout << name << "." << fn_name << std::endl;
  std::cout << name << ".name(): " << mat.name() << std::endl;
  std::cout << name << ".active_name(): " << mat.active_name() << std::endl;
  std::cout << name << ".active_index(): " << mat.active_index() << std::endl;
  std::cout << std::endl;
}

int main() {
  Morpheus::initialize();
  {
    Morpheus::DynamicMatrix<double, int, Kokkos::Serial> A;

    stats(A, "A", "resize(5, 10, 15)");
    A.resize(5, 10, 15);

    try {
      A.resize(5, 10, 15, 20);
    } catch (Morpheus::RuntimeException& e) {
      std::cerr << "Exception Raised:: " << e.what() << std::endl;
    }
    stats(A, "A", "resize(5, 10, 15, 20)");

    A = Morpheus::CsrMatrix<double, int, Kokkos::Serial>();
    stats(A, "A", "resize(5, 10, 15)");
    A.resize(5, 10, 15);

    A = Morpheus::DiaMatrix<double, int, Kokkos::Serial>();
    stats(A, "A", "resize(5, 10, 15, 20)");
    A.resize(5, 10, 15, 20);
    A.resize(5, 10, 15, 20, 35);
    stats(A, "A", "resize(5, 10, 15, 20, 35)");
  }
  Morpheus::finalize();
  return 0;
}