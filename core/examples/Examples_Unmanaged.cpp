/**
 * Exampels_Unamanged.cpp
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

using index_type   = int;
using value_type   = double;
using memory_trait = Kokkos::MemoryUnmanaged;

using vec = Morpheus::DenseVector<value_type, memory_trait>;

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    index_type n  = 15;
    value_type* v = (value_type*)malloc(n * sizeof(value_type));

    for (index_type i = 0; i < n; i++) {
      v[i] = i * n;
    }

    {
      vec x("x", n, v);
      typename vec::HostMirror xm = Morpheus::create_mirror_container(x);
      Morpheus::print(xm);

      xm(5) = -15;
    }

    for (index_type i = 0; i < n; i++) {
      std::cout << "\t [" << i << "] : " << v[i] << std::endl;
    }

    free(v);
  }
  Morpheus::finalize();

  return 0;
}