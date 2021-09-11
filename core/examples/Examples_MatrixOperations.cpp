/**
 * Examples_MatrixOperations.cpp
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
#include <sstream>

using value_type = double;
using index_type = int;

using coo_host = Morpheus::CooMatrix<value_type, index_type, Kokkos::HostSpace>;
using csr_host = Morpheus::CsrMatrix<value_type, index_type, Kokkos::HostSpace>;
using dia_host = Morpheus::DiaMatrix<value_type, index_type, Kokkos::HostSpace>;
using vec_host = Morpheus::DenseVector<value_type, Kokkos::HostSpace>;
using dynamic_host =
    Morpheus::DynamicMatrix<value_type, index_type, Kokkos::HostSpace>;

#if defined(MORPHEUS_ENABLE_CUDA)
using coo_dev = Morpheus::CooMatrix<value_type, index_type, Kokkos::CudaSpace>;
using csr_dev = Morpheus::CsrMatrix<value_type, index_type, Kokkos::CudaSpace>;
using dia_dev = Morpheus::DiaMatrix<value_type, index_type, Kokkos::CudaSpace>;
using vec_dev = Morpheus::DenseVector<value_type, Kokkos::CudaSpace>;
using dynamic_dev =
    Morpheus::DynamicMatrix<value_type, index_type, Kokkos::CudaSpace>;
#endif

const value_type UPDATE_VAL = -5;

//    [10  0 20]
//    [ 0  0  0]
//    [ 0  0 30]
//    [40 50 60]

template <typename Matrix>
void init(Matrix& A, Morpheus::CooTag) {
  Matrix mat(4, 3, 6);

  // initialize matrix entries
  mat.row_indices[0]    = 0;
  mat.column_indices[0] = 0;
  mat.values[0]         = 10;
  mat.row_indices[1]    = 0;
  mat.column_indices[1] = 2;
  mat.values[1]         = 20;
  mat.row_indices[2]    = 2;
  mat.column_indices[2] = 2;
  mat.values[2]         = 30;
  mat.row_indices[3]    = 3;
  mat.column_indices[3] = 0;
  mat.values[3]         = 40;
  mat.row_indices[4]    = 3;
  mat.column_indices[4] = 1;
  mat.values[4]         = 50;
  mat.row_indices[5]    = 3;
  mat.column_indices[5] = 2;
  mat.values[5]         = 60;

  A = mat;
}

template <typename Matrix>
void init(Matrix& A, Morpheus::CsrTag) {
  Matrix mat(4, 3, 6);

  // initialize matrix entries
  mat.row_offsets[0] = 0;
  mat.row_offsets[1] = 2;
  mat.row_offsets[2] = 2;
  mat.row_offsets[3] = 3;
  mat.row_offsets[4] = 6;

  mat.column_indices[0] = 0;
  mat.values[0]         = 10;
  mat.column_indices[1] = 2;
  mat.values[1]         = 20;
  mat.column_indices[2] = 2;
  mat.values[2]         = 30;
  mat.column_indices[3] = 0;
  mat.values[3]         = 40;
  mat.column_indices[4] = 1;
  mat.values[4]         = 50;
  mat.column_indices[5] = 2;
  mat.values[5]         = 60;

  A = mat;
}

template <typename Matrix>
void init(Matrix& A, Morpheus::DiaTag) {
  Matrix mat(4, 3, 6, 5);

  // Diagonal offsets
  mat.diagonal_offsets[0] = -3;
  mat.diagonal_offsets[1] = -2;
  mat.diagonal_offsets[2] = -1;
  mat.diagonal_offsets[3] = 0;
  mat.diagonal_offsets[4] = 2;

  // First Diagonal
  mat.values(0, 0) = -1;
  mat.values(1, 0) = -1;
  mat.values(2, 0) = -1;
  mat.values(3, 0) = 40;
  // Second Diagonal
  mat.values(0, 1) = -2;
  mat.values(1, 1) = -2;
  mat.values(2, 1) = 0;
  mat.values(3, 1) = 50;
  // Third Diagonal
  mat.values(0, 2) = -3;
  mat.values(1, 2) = 0;
  mat.values(2, 2) = 0;
  mat.values(3, 2) = 60;
  // Fourth Diagonal
  mat.values(0, 3) = 10;
  mat.values(1, 3) = 0;
  mat.values(2, 3) = 30;
  mat.values(3, 3) = -4;
  // Fifth Diagonal
  mat.values(0, 4) = 20;
  mat.values(1, 4) = -5;
  mat.values(2, 4) = -5;
  mat.values(3, 4) = -5;

  A = mat;
}

template <typename Matrix>
void ref(Matrix& A, Morpheus::CooTag) {
  Matrix mat(4, 3, 6);

  // initialize matrix entries
  mat.row_indices[0]    = 0;
  mat.column_indices[0] = 0;
  mat.values[0]         = UPDATE_VAL;
  mat.row_indices[1]    = 0;
  mat.column_indices[1] = 2;
  mat.values[1]         = 20;
  mat.row_indices[2]    = 2;
  mat.column_indices[2] = 2;
  mat.values[2]         = UPDATE_VAL;
  mat.row_indices[3]    = 3;
  mat.column_indices[3] = 0;
  mat.values[3]         = 40;
  mat.row_indices[4]    = 3;
  mat.column_indices[4] = 1;
  mat.values[4]         = 50;
  mat.row_indices[5]    = 3;
  mat.column_indices[5] = 2;
  mat.values[5]         = 60;

  A = mat;
}

template <typename Matrix>
void ref(Matrix& A, Morpheus::CsrTag) {
  Matrix mat(4, 3, 6);

  // initialize matrix entries
  mat.row_offsets[0] = 0;
  mat.row_offsets[1] = 2;
  mat.row_offsets[2] = 2;
  mat.row_offsets[3] = 3;
  mat.row_offsets[4] = 6;

  mat.column_indices[0] = 0;
  mat.values[0]         = UPDATE_VAL;
  mat.column_indices[1] = 2;
  mat.values[1]         = 20;
  mat.column_indices[2] = 2;
  mat.values[2]         = UPDATE_VAL;
  mat.column_indices[3] = 0;
  mat.values[3]         = 40;
  mat.column_indices[4] = 1;
  mat.values[4]         = 50;
  mat.column_indices[5] = 2;
  mat.values[5]         = 60;

  A = mat;
}

template <typename Matrix>
void ref(Matrix& A, Morpheus::DiaTag) {
  Matrix mat(4, 3, 6, 5);

  // Diagonal offsets
  mat.diagonal_offsets[0] = -3;
  mat.diagonal_offsets[1] = -2;
  mat.diagonal_offsets[2] = -1;
  mat.diagonal_offsets[3] = 0;
  mat.diagonal_offsets[4] = 2;

  // First Diagonal
  mat.values(0, 0) = -1;
  mat.values(1, 0) = -1;
  mat.values(2, 0) = -1;
  mat.values(3, 0) = 40;
  // Second Diagonal
  mat.values(0, 1) = -2;
  mat.values(1, 1) = -2;
  mat.values(2, 1) = 0;
  mat.values(3, 1) = 50;
  // Third Diagonal
  mat.values(0, 2) = -3;
  mat.values(1, 2) = 0;
  mat.values(2, 2) = 0;
  mat.values(3, 2) = 60;
  // Fourth Diagonal
  mat.values(0, 3) = UPDATE_VAL;
  mat.values(1, 3) = 0;
  mat.values(2, 3) = UPDATE_VAL;
  mat.values(3, 3) = -4;
  // Fifth Diagonal
  mat.values(0, 4) = 20;
  mat.values(1, 4) = -5;
  mat.values(2, 4) = -5;
  mat.values(3, 4) = -5;

  A = mat;
}

template <typename Matrix>
void validate(const Matrix& A, const Matrix& Aref, Morpheus::CooTag) {
  for (index_type n = 0; n < A.nnnz(); n++) {
    if (A.row_indices[n] != Aref.row_indices[n]) {
      std::stringstream msg;
      msg << "Row Indices at index " << n << " differ: " << A.row_indices[n]
          << " != " << Aref.row_indices[n] << "\n";
      throw Morpheus::RuntimeException(msg.str());
    }

    if (A.column_indices[n] != Aref.column_indices[n]) {
      std::stringstream msg;
      msg << "Column Indices at index " << n
          << " differ: " << A.column_indices[n]
          << " != " << Aref.column_indices[n] << "\n";
      throw Morpheus::RuntimeException(msg.str());
    }

    if (A.values[n] != Aref.values[n]) {
      std::stringstream msg;
      msg << "Values at index " << n << " differ: " << A.values[n]
          << " != " << Aref.values[n] << "\n";
      throw Morpheus::RuntimeException(msg.str());
    }
  }
}

template <typename Matrix>
void validate(const Matrix& A, const Matrix& Aref, Morpheus::CsrTag) {
  for (index_type i = 0; i < A.nrows(); i++) {
    if (A.row_offsets[i] != Aref.row_offsets[i]) {
      std::stringstream msg;
      msg << "Row Offsets at index " << i << " differ: " << A.row_offsets[i]
          << " != " << Aref.row_offsets[i] << "\n";
      throw Morpheus::RuntimeException(msg.str());
    }
    for (index_type jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++) {
      if (A.column_indices[jj] != Aref.column_indices[jj]) {
        std::stringstream msg;
        msg << "Column Indices at index " << jj
            << " differ: " << A.column_indices[jj]
            << " != " << Aref.column_indices[jj] << "\n";
        throw Morpheus::RuntimeException(msg.str());
      }

      if (A.values[jj] != Aref.values[jj]) {
        std::stringstream msg;
        msg << "Values at index " << jj << " differ: " << A.values[jj]
            << " != " << Aref.values[jj] << "\n";
        throw Morpheus::RuntimeException(msg.str());
      }
    }
  }
}

template <typename Matrix>
void validate(const Matrix& A, const Matrix& Aref, Morpheus::DiaTag) {
  const index_type ndiag = A.values.ncols();

  for (index_type row = 0; row < A.nrows(); row++) {
    for (index_type n = 0; n < ndiag; n++) {
      if (A.diagonal_offsets[n] != Aref.diagonal_offsets[n]) {
        std::stringstream msg;
        msg << "Diagonal Offsets at index " << n
            << " differ: " << A.diagonal_offsets[n]
            << " != " << Aref.diagonal_offsets[n] << "\n";
        throw Morpheus::RuntimeException(msg.str());
      }

      const index_type col = row + A.diagonal_offsets[n];

      if ((col >= 0 && col < A.ncols()) && (col == row)) {
        if (A.values(row, n) != Aref.values(row, n)) {
          std::stringstream msg;
          msg << "Values at index (" << row << "," << n
              << ") differ: " << A.values(row, n)
              << " != " << Aref.values(row, n) << "\n";
          throw Morpheus::RuntimeException(msg.str());
        }
      }
    }
  }
}

template <typename DynamicMatrix, typename Matrix, typename Vector,
          typename ExecSpace>
void update_and_print(DynamicMatrix& Ad) {
  using tag = typename Matrix::HostMirror::tag;
  typename Matrix::HostMirror Ah, Ah_ref, Aout;

  init(Ah, tag{});
  ref(Ah_ref, tag{});

  Vector diag(Ah.ncols(), UPDATE_VAL);

  Morpheus::copy(Ah, Ad);
  Morpheus::update_diagonal<ExecSpace>(Ad, diag);
  Morpheus::copy(Ad, Aout);

  validate(Aout, Ah_ref, tag{});
}

template <typename Matrix, typename Vector, typename ExecSpace>
void update_and_print() {
  using tag = typename Matrix::HostMirror::tag;
  typename Matrix::HostMirror Ah, Ah_ref, Aout;

  init(Ah, tag{});
  ref(Ah_ref, tag{});

  Vector diag(Ah.ncols(), UPDATE_VAL);

  Matrix A;
  Morpheus::copy(Ah, A);
  Morpheus::update_diagonal<ExecSpace>(A, diag);
  Morpheus::copy(A, Aout);

  validate(Aout, Ah_ref, tag{});
}

int main() {
  Morpheus::initialize();
  {
    {
      dynamic_host A;

      update_and_print<coo_host, vec_host, Kokkos::Serial>();
      update_and_print<csr_host, vec_host, Kokkos::Serial>();
      update_and_print<dia_host, vec_host, Kokkos::Serial>();

      update_and_print<coo_host, vec_host, Morpheus::Serial>();
      update_and_print<csr_host, vec_host, Morpheus::Serial>();
      update_and_print<dia_host, vec_host, Morpheus::Serial>();

      update_and_print<dynamic_host, coo_host, vec_host, Kokkos::Serial>(A);
      update_and_print<dynamic_host, csr_host, vec_host, Kokkos::Serial>(A);
      update_and_print<dynamic_host, dia_host, vec_host, Kokkos::Serial>(A);

      update_and_print<dynamic_host, coo_host, vec_host, Morpheus::Serial>(A);
      update_and_print<dynamic_host, csr_host, vec_host, Morpheus::Serial>(A);
      update_and_print<dynamic_host, dia_host, vec_host, Morpheus::Serial>(A);
    }

#if defined(MORPHEUS_ENABLE_OPENMP)
    {
      dynamic_host A;
      update_and_print<coo_host, vec_host, Kokkos::OpenMP>();
      update_and_print<csr_host, vec_host, Kokkos::OpenMP>();
      update_and_print<dia_host, vec_host, Kokkos::OpenMP>();

      update_and_print<coo_host, vec_host, Morpheus::OpenMP>();
      update_and_print<csr_host, vec_host, Morpheus::OpenMP>();
      update_and_print<dia_host, vec_host, Morpheus::OpenMP>();

      update_and_print<dynamic_host, coo_host, vec_host, Kokkos::OpenMP>(A);
      update_and_print<dynamic_host, csr_host, vec_host, Kokkos::OpenMP>(A);
      update_and_print<dynamic_host, dia_host, vec_host, Kokkos::OpenMP>(A);

      update_and_print<dynamic_host, coo_host, vec_host, Morpheus::OpenMP>(A);
      update_and_print<dynamic_host, csr_host, vec_host, Morpheus::OpenMP>(A);
      update_and_print<dynamic_host, dia_host, vec_host, Morpheus::OpenMP>(A);
    }
#endif

#if defined(MORPHEUS_ENABLE_CUDA)
    {
      dynamic_dev A;

      update_and_print<coo_dev, vec_dev, Kokkos::Cuda>();
      update_and_print<csr_dev, vec_dev, Kokkos::Cuda>();
      update_and_print<dia_dev, vec_dev, Kokkos::Cuda>();

      update_and_print<coo_dev, vec_dev, Morpheus::Cuda>();
      update_and_print<csr_dev, vec_dev, Morpheus::Cuda>();
      update_and_print<dia_dev, vec_dev, Morpheus::Cuda>();

      update_and_print<dynamic_dev, coo_dev, vec_dev, Kokkos::Cuda>(A);
      update_and_print<dynamic_dev, csr_dev, vec_dev, Kokkos::Cuda>(A);
      update_and_print<dynamic_dev, dia_dev, vec_dev, Kokkos::Cuda>(A);

      update_and_print<dynamic_dev, coo_dev, vec_dev, Morpheus::Cuda>(A);
      update_and_print<dynamic_dev, csr_dev, vec_dev, Morpheus::Cuda>(A);
      update_and_print<dynamic_dev, dia_dev, vec_dev, Morpheus::Cuda>(A);
    }
#endif
  }
  Morpheus::finalize();
}