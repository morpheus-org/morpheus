/**
 * Examples_VectorSlice_copy.cpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2022 The University of Edinburgh
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

using vec    = Morpheus::DenseVector<double, Kokkos::HostSpace>;
using serial = Kokkos::Serial;

int main() {
  Morpheus::initialize();
  {
    vec x(50, 2), y(50, 0);

    Morpheus::copy(x, y, 0, 10);

    x[3] = -15;
    Morpheus::print(x);
    Morpheus::print(y);
  }
  Morpheus::finalize();
}