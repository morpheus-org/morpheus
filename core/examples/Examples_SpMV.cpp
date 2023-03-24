/**
 * Examples_SpMV.cpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2023 The University of Edinburgh
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

/* Default execution space: Either Serial or OpenMP backend.
 * If compiled with GPU support this is either HIP or CUDA. */
using Space = Morpheus::DefaultExecutionSpace;
// A random number generator running in default execution space
using Generator =
    Kokkos::Random_XorShift64_Pool<typename Space::execution_space>;
/* A Dynamic Matrix holding values of type double
 * and lives in the memory space of default execution space */
using Matrix = Morpheus::DynamicMatrix<double, Space>;
/* A Dense Vector holding values of type double
 * and lives in the memory space of default execution space */
using Vector = Morpheus::DenseVector<double, Space>;

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    std::string filename = argv[1];
    // Read format ID from command-line
    int fmt_id = atoi(argv[2]);

    // Load matrix on host
    typename Matrix::HostMirror Ah;
    Morpheus::IO::read_matrix_market_file(Ah, filename);

    /* In-place convert matrix to a dynamic matrix
     * with its active state set as per fmt_id */
    Morpheus::convert<Morpheus::Serial>(Ah, fmt_id);

    // Create a dynamic matrix that resides in Space
    Matrix A = Morpheus::create_mirror_container<Space>(Ah);
    // Copy data from host to container in Space
    Morpheus::copy(Ah, A);

    // Randomly initialize x and set y to zero
    Vector x(Ah.ncols(), Generator(0), 0, 1);
    Vector y(Ah.nrows(), 0);

    // SpMV multiplication in Space
    Morpheus::multiply<Space>(A, x, y);
  }
  Morpheus::finalize();
}