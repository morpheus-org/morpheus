/**
 * DenseMatrix.cpp
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

#include <morpheus/core/core.hpp>
#include <morpheus/containers/dense_matrix.hpp>

using Matrix =
    Morpheus::DenseMatrix<double, int, Kokkos::LayoutRight, Kokkos::Serial>;

template <typename M>
void print_stats(const M& A) {
  std::cout << A.name() << ":\t" << A.nrows() << "\t" << A.ncols() << "\t"
            << A.nnnz() << std::endl;

  std::cout << "\tData:" << std::endl;
  for (int i = 0; i < A.nrows(); i++) {
    std::cout << "\t\t";
    for (int j = 0; j < A.ncols(); j++) {
      std::cout << A(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);

  {
    Matrix A;
    print_stats(A);
  }

  {
    Matrix A(2, 3);
    print_stats(A);
  }

  {
    Matrix A(2, 3, 4.0);
    print_stats(A);
  }

  {
    Matrix A("A", 2, 3);
    print_stats(A);
  }

  {
    Matrix A("A", 2, 3, 3.0);
    print_stats(A);
  }

  {
    Matrix A("A", 2, 3, 3.0);
    A.resize(1, 2);
    print_stats(A);
  }

  {
    Matrix A("A", 2, 3, 3.0);
    A.resize(4, 3);
    print_stats(A);
  }

  {
    Matrix A("A", 2, 3);

    for (int i = 0; i < A.nrows(); i++) {
      for (int j = 0; j < A.ncols(); j++) {
        A(i, j) = i * A.ncols() + j;
      }
    }

    print_stats(A);
  }

  {
    Morpheus::DenseMatrix<double, int, Kokkos::LayoutLeft, Kokkos::Serial> A(
        "A", 2, 3);

    for (int i = 0; i < A.nrows(); i++) {
      for (int j = 0; j < A.ncols(); j++) {
        A(i, j) = i * A.ncols() + j;
      }
    }

    print_stats(A);
  }

  Morpheus::finalize();
}