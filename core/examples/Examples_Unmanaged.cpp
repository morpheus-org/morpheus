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

using vec    = Morpheus::DenseVector<value_type, memory_trait>;
using matrix = Morpheus::DenseMatrix<value_type, memory_trait>;
using coo    = Morpheus::CooMatrix<value_type, memory_trait>;
using csr    = Morpheus::CsrMatrix<value_type, memory_trait>;

int main(int argc, char* argv[]) {
  Morpheus::initialize(argc, argv);
  {
    // DenseVector unmanaged
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
    // DenseMatrix unmanaged
    {
      index_type n    = 5;
      value_type* mat = (value_type*)malloc(n * n * sizeof(value_type*));

      for (index_type i = 0; i < n; i++) {
        for (index_type j = 0; j < n; j++) {
          mat[i * n + j] = i * n + j;
        }
      }

      {
        matrix m("m", n, n, mat);
        typename matrix::HostMirror mm = Morpheus::create_mirror_container(m);
        Morpheus::print(mm);

        mm(2, 2) = -15;
      }

      for (index_type i = 0; i < n; i++) {
        for (index_type j = 0; j < n; j++) {
          std::cout << mat[i * n + j] << "\t";
        }
        std::cout << std::endl;
      }

      free(mat);
    }
    // CooMatrix unmanaged
    {
      index_type M = 4, N = 3, NNZ = 6;
      value_type* vals = (value_type*)malloc(NNZ * sizeof(value_type));
      index_type* rind = (index_type*)malloc(NNZ * sizeof(index_type));
      index_type* cind = (index_type*)malloc(NNZ * sizeof(index_type));

      rind[0] = 0;
      cind[0] = 0;
      vals[0] = 10;
      rind[1] = 0;
      cind[1] = 2;
      vals[1] = 20;
      rind[2] = 2;
      cind[2] = 2;
      vals[2] = 30;
      rind[3] = 3;
      cind[3] = 0;
      vals[3] = 40;
      rind[4] = 3;
      cind[4] = 1;
      vals[4] = 50;
      rind[5] = 3;
      cind[5] = 2;
      vals[5] = 60;

      {
        coo A("A", M, N, NNZ, rind, cind, vals);
        typename coo::HostMirror Am = Morpheus::create_mirror_container(A);
        Morpheus::print(A);

        A.row_indices(2) = -15;
        A.values(5)      = -15;
      }

      for (index_type i = 0; i < NNZ; i++) {
        std::cout << "\t [" << i << "] : (" << rind[i] << ", " << cind[i]
                  << ", " << vals[i] << ")" << std::endl;
      }

      free(vals);
      free(rind);
      free(cind);
    }
    // CsrMatrix unmanaged
    {
      index_type M = 4, N = 3, NNZ = 6;
      value_type* vals = (value_type*)malloc(NNZ * sizeof(value_type));
      index_type* roff = (index_type*)malloc((M + 1) * sizeof(index_type));
      index_type* cind = (index_type*)malloc(NNZ * sizeof(index_type));

      roff[0] = 0;
      roff[1] = 2;
      roff[2] = 3;
      roff[3] = 5;
      roff[4] = 6;

      cind[0] = 0;
      cind[1] = 1;
      cind[2] = 2;
      cind[3] = 0;
      cind[4] = 2;
      cind[5] = 1;

      vals[0] = 10;
      vals[1] = 20;
      vals[2] = 30;
      vals[3] = 40;
      vals[4] = 50;
      vals[5] = 60;

      {
        csr A("A", M, N, NNZ, roff, cind, vals);
        typename csr::HostMirror Am = Morpheus::create_mirror_container(A);
        Morpheus::print(A);

        A.row_offsets(2) = -15;
        A.values(5)      = -15;
      }

      for (index_type i = 0; i < M; i++) {
        std::cout << "\t [" << i << "] : " << roff[i] << std::endl;
      }
      std::cout << std::endl;
      for (index_type i = 0; i < NNZ; i++) {
        std::cout << "\t [" << i << "] : (" << cind[i] << ", " << vals[i] << ")"
                  << std::endl;
      }

      free(vals);
      free(roff);
      free(cind);
    }
  }
  Morpheus::finalize();

  return 0;
}