/**
 * MatrixGenerator.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2023 The University of Edinburgh
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

#ifndef TEST_CORE_MATRIX_GENERATOR_HPP
#define TEST_CORE_MATRIX_GENERATOR_HPP

#include <gtest/gtest.h>
#include <Morpheus_Core.hpp>
#include <random>

#include <utils/Macros_DenseMatrix.hpp>

namespace Morpheus {
namespace Test {
template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
class DiagonalMatrixGenerator {
 public:
  using DenseMatrix =
      typename Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout,
                                     Space>::HostMirror;
  using SparseMatrix =
      typename Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout,
                                   Space>::HostMirror;
  using size_type = typename SparseMatrix::size_type;

  DiagonalMatrixGenerator(size_type num_rows, size_type num_cols,
                          std::vector<int>& diag_indexes)
      : nrows(num_rows),
        ncols(num_cols),
        diags(diag_indexes),
        dense_mat(num_rows, num_cols, 0) {}

  DenseMatrix& generate() {
    for (int i = 0; i < (int)nrows; i++) {
      for (int j = 0; j < (int)ncols; j++) {
        for (size_type d = 0; d < diags.size(); d++) {
          if (abs(j - i) == abs((int)diags[d])) {
            dense_mat(i, j) =
                -2.0 + (((ValueType)rand() / RAND_MAX) * (2.0 - (-2.0)));
          }
        }
      }
    }
    return dense_mat;
  }

  void generate(SparseMatrix& mat) {
    if (is_empty_container(dense_mat)) {
      dense_mat = generate();
    }
    Morpheus::convert<Morpheus::Serial>(dense_mat, mat);
  }

  size_type nrows, ncols;
  std::vector<int> diags;
  DenseMatrix dense_mat;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
class SparseDiagonalMatrixGenerator {
 public:
  using DenseMatrix =
      typename Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout,
                                     Space>::HostMirror;
  using SparseMatrix =
      typename Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout,
                                   Space>::HostMirror;
  using size_type = typename SparseMatrix::size_type;

  SparseDiagonalMatrixGenerator(size_type num_rows, size_type num_cols,
                                std::vector<int>& diag_indexes)
      : nrows(num_rows),
        ncols(num_cols),
        diags(diag_indexes),
        dense_mat(num_rows, num_cols, 0) {}

  DenseMatrix& generate() {
    for (int i = 0; i < (int)nrows; i++) {
      for (int j = 0; j < (int)ncols; j++) {
        for (size_type d = 0; d < diags.size(); d++) {
          if (abs(j - i) == abs((int)diags[d])) {
            // If main diagonal, add an entry every five rows
            if ((diags[d] == 0 && i % 5) || (diags[d] != 0)) {
              dense_mat(i, j) =
                  -2.0 + (((ValueType)rand() / RAND_MAX) * (2.0 - (-2.0)));
            }
          }
        }
      }
    }
    return dense_mat;
  }

  void generate(SparseMatrix& mat) {
    if (is_empty_container(dense_mat)) {
      dense_mat = generate();
    }
    Morpheus::convert<Morpheus::Serial>(dense_mat, mat);
  }

  size_type nrows, ncols;
  std::vector<int> diags;
  DenseMatrix dense_mat;
};

template <typename ValueType, typename IndexType, typename ArrayLayout,
          typename Space>
class AntiDiagonalMatrixGenerator {
 public:
  using DenseMatrix =
      typename Morpheus::DenseMatrix<ValueType, IndexType, ArrayLayout,
                                     Space>::HostMirror;
  using SparseMatrix =
      typename Morpheus::CooMatrix<ValueType, IndexType, ArrayLayout,
                                   Space>::HostMirror;
  using size_type = typename SparseMatrix::size_type;

  AntiDiagonalMatrixGenerator(size_type num_rows, size_type num_cols,
                              std::vector<int>& diag_indexes)
      : nrows(num_rows),
        ncols(num_cols),
        diags(diag_indexes),
        dense_mat(num_rows, num_cols, 0) {}

  DenseMatrix& generate() {
    for (int i = 0; i < (int)nrows; i++) {
      for (int j = 0; j < (int)ncols; j++) {
        for (size_type d = 0; d < diags.size(); d++) {
          if (abs(j - i) == abs((int)diags[d])) {
            dense_mat(i, ncols - j - 1) =
                -2.0 + (((ValueType)rand() / RAND_MAX) * (2.0 - (-2.0)));
          }
        }
      }
    }

    return dense_mat;
  }

  void generate(SparseMatrix& mat) {
    if (is_empty_container(dense_mat)) {
      dense_mat = generate();
    }

    Morpheus::convert<Morpheus::Serial>(dense_mat, mat);
  }

  size_type nrows, ncols;
  std::vector<int> diags;
  DenseMatrix dense_mat;
};
}  // namespace Test
}  // namespace Morpheus

#endif  // TEST_CORE_MATRIX_GENERATOR_HPP