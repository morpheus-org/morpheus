/**
 * sorting.cpp
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

#include <morpheus/containers/coo_matrix.hpp>
#include <morpheus/algorithms/sort.hpp>
#include <morpheus/algorithms/copy.hpp>
#include <morpheus/algorithms/print.hpp>
#include <Kokkos_Random.hpp>

using coo  = Morpheus::CooMatrix<double, int, Kokkos::Serial>;
using ivec = typename coo::index_array_type;
using vvec = typename coo::value_array_type;

int main() {
  Morpheus::initialize();
  {
    using Generator = Kokkos::Random_XorShift64_Pool<Kokkos::Serial>;

    vvec vv("vv", 20, Generator(0), 0, 100);
    ivec ii("ii", 20, Generator(0), 0, 100), jj("jj", 20, Generator(1), 0, 100);
    coo A("A", ii, jj, vv), B;
    Morpheus::copy(A, B);

    Morpheus::print(A);
    std::cout << "Is sorted:: " << Morpheus::is_sorted(A) << std::endl;
    Morpheus::sort_by_row_and_column(A);
    Morpheus::print(A);
    std::cout << "Is sorted:: " << Morpheus::is_sorted(A) << std::endl;
    std::cout << "Is sorted:: " << B.is_sorted() << std::endl;
    B.sort_by_row_and_column();
    std::cout << "Is sorted:: " << B.is_sorted() << std::endl;
  }
  Morpheus::finalize();
}