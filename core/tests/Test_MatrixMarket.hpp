/**
 * Test_MatrixMarket.hpp
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

#ifndef TEST_CORE_TEST_MATRIXMARKET_HPP
#define TEST_CORE_TEST_MATRIXMARKET_HPP

#include <Morpheus_Core.hpp>
#include <utils/Macros_CooMatrix.hpp>
#include <utils/Macros_DynamicMatrix.hpp>
#include <utils/Macros_DenseMatrix.hpp>
#include <utils/Macros_DenseVector.hpp>

#include <string>
#include <iostream>
#include <filesystem>

namespace Test {

std::string get_mm_test_path() {
  std::string file_path = __FILE__;
  return file_path.substr(0, file_path.rfind("/")) + "/mm_data";
}

// void create_mm_out_path() {
//   std::string mmpath("mm_output");

//   if (!std::filesystem::exists(mmpath)) {
//     std::filesystem::create_directories(mmpath);
//   }
// }

std::string get_mm_out_path() { return "mm_output/"; }

TEST(MatrixMarket, ReadMatrixComplex) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/complex.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, ReadMatrixHermitian) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/hermitian.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, ReadMatrixSkewSymmetric) {
  Morpheus::CooMatrix<int, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/skew-symmetric.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, ReadMatrixSymmetricReal) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A, Aref(4, 4, 7);
  std::string filename(get_mm_test_path() + "/symmetric_real.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 1;
  Aref.values(0)         = 2.123;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 2;
  Aref.values(1)         = -45.345;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 2.123;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 2;
  Aref.values(3)         = -4.8765;
  Aref.row_indices(4)    = 2;
  Aref.column_indices(4) = 0;
  Aref.values(4)         = -45.345;
  Aref.row_indices(5)    = 2;
  Aref.column_indices(5) = 1;
  Aref.values(5)         = -4.8765;
  Aref.row_indices(6)    = 3;
  Aref.column_indices(6) = 3;
  Aref.values(6)         = 1.15;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixSymmetricInteger) {
  Morpheus::CooMatrix<int, Morpheus::HostSpace> A, Aref(4, 4, 7);
  std::string filename(get_mm_test_path() + "/symmetric_integer.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 1;
  Aref.values(0)         = 2;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 2;
  Aref.values(1)         = -45;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 2;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 2;
  Aref.values(3)         = -4;
  Aref.row_indices(4)    = 2;
  Aref.column_indices(4) = 0;
  Aref.values(4)         = -45;
  Aref.row_indices(5)    = 2;
  Aref.column_indices(5) = 1;
  Aref.values(5)         = -4;
  Aref.row_indices(6)    = 3;
  Aref.column_indices(6) = 3;
  Aref.values(6)         = 1;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixSymmetricPattern) {
  Morpheus::CooMatrix<int, Morpheus::HostSpace> A, Aref(4, 4, 7);
  std::string filename(get_mm_test_path() + "/symmetric_pattern.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 1;
  Aref.values(0)         = 1;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 2;
  Aref.values(1)         = 1;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 1;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 2;
  Aref.values(3)         = 1;
  Aref.row_indices(4)    = 2;
  Aref.column_indices(4) = 0;
  Aref.values(4)         = 1;
  Aref.row_indices(5)    = 2;
  Aref.column_indices(5) = 1;
  Aref.values(5)         = 1;
  Aref.row_indices(6)    = 3;
  Aref.column_indices(6) = 3;
  Aref.values(6)         = 1;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricReal) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A, Aref(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_real.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 0;
  Aref.values(0)         = 1.11;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 1;
  Aref.values(1)         = 2.22;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 3.33;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 1;
  Aref.values(3)         = 4.44;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricRealUnsorted) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A, Asorted(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_real_unsorted.mtx");

  Asorted.row_indices(0)    = 0;
  Asorted.column_indices(0) = 0;
  Asorted.values(0)         = 1.11;
  Asorted.row_indices(1)    = 0;
  Asorted.column_indices(1) = 1;
  Asorted.values(1)         = 2.22;
  Asorted.row_indices(2)    = 1;
  Asorted.column_indices(2) = 0;
  Asorted.values(2)         = 3.33;
  Asorted.row_indices(3)    = 1;
  Asorted.column_indices(3) = 1;
  Asorted.values(3)         = 4.44;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Asorted);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Asorted));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricRealDynamic) {
  Morpheus::DynamicMatrix<double, Morpheus::HostSpace> A, Asorted;

  Morpheus::CooMatrix<double, Morpheus::HostSpace> Acoo_sorted(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_real_unsorted.mtx");

  Acoo_sorted.row_indices(0)    = 0;
  Acoo_sorted.column_indices(0) = 0;
  Acoo_sorted.values(0)         = 1.11;
  Acoo_sorted.row_indices(1)    = 0;
  Acoo_sorted.column_indices(1) = 1;
  Acoo_sorted.values(1)         = 2.22;
  Acoo_sorted.row_indices(2)    = 1;
  Acoo_sorted.column_indices(2) = 0;
  Acoo_sorted.values(2)         = 3.33;
  Acoo_sorted.row_indices(3)    = 1;
  Acoo_sorted.column_indices(3) = 1;
  Acoo_sorted.values(3)         = 4.44;

  Asorted = Acoo_sorted;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_DYNAMIC_CONTAINERS(A, Asorted);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Asorted));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricRealDynamicWrongState) {
  Morpheus::DynamicMatrix<double, Morpheus::HostSpace> A;

  A.activate(Morpheus::CSR_FORMAT);

  std::string filename(get_mm_test_path() + "/unsymmetric_real_unsorted.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::RuntimeException);
}

TEST(MatrixMarket, ReadMatrixUnsymmetricFloat) {
  Morpheus::CooMatrix<float, Morpheus::HostSpace> A, Aref(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_real.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 0;
  Aref.values(0)         = 1.11;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 1;
  Aref.values(1)         = 2.22;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 3.33;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 1;
  Aref.values(3)         = 4.44;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricInteger) {
  Morpheus::CooMatrix<int, Morpheus::HostSpace> A, Aref(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_integer.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 0;
  Aref.values(0)         = 1;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 1;
  Aref.values(1)         = 2;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 3;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 1;
  Aref.values(3)         = 4;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricLongLong) {
  Morpheus::CooMatrix<long long, Morpheus::HostSpace> A, Aref(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_integer.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 0;
  Aref.values(0)         = 1;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 1;
  Aref.values(1)         = 2;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 3;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 1;
  Aref.values(3)         = 4;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixUnsymmetricPattern) {
  Morpheus::CooMatrix<int, Morpheus::HostSpace> A, Aref(2, 2, 4);
  std::string filename(get_mm_test_path() + "/unsymmetric_pattern.mtx");

  Aref.row_indices(0)    = 0;
  Aref.column_indices(0) = 0;
  Aref.values(0)         = 1;
  Aref.row_indices(1)    = 0;
  Aref.column_indices(1) = 1;
  Aref.values(1)         = 1;
  Aref.row_indices(2)    = 1;
  Aref.column_indices(2) = 0;
  Aref.values(2)         = 1;
  Aref.row_indices(3)    = 1;
  Aref.column_indices(3) = 1;
  Aref.values(3)         = 1;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

TEST(MatrixMarket, ReadMatrixInvalidRowIndexZeroBased) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_row_zero_index.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidColumnIndexZeroBased) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() +
                       "/mat_invalid_column_zero_index.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidRowIndexOverRows) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_row_over_index.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidColumnIndexOverCols) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() +
                       "/mat_invalid_column_over_index.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidDataType) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_data_type.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixUnexpectedEOF) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_unexpected_eof.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidBanner) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_banner.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidStorageFormat) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_storage_format.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidSymmetry) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_symmetry.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadMatrixInvalidSize) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/mat_invalid_size.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadArrayUnsymmetricIntegerDenseMatrix) {
  Morpheus::DenseMatrix<int, Morpheus::HostSpace> A, Aref(2, 2);
  std::string filename(get_mm_test_path() + "/array_unsymmetric_integer.mtx");

  Aref(0, 0) = 1;
  Aref(0, 1) = -2;
  Aref(1, 0) = 3;
  Aref(1, 1) = -4;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(Aref, A));
}

TEST(MatrixMarket, ReadArrayUnsymmetricIntegerDenseVector) {
  Morpheus::DenseVector<int, Morpheus::HostSpace> v, vref(4);
  std::string filename(get_mm_test_path() + "/vec_unsymmetric_integer.mtx");

  vref(0) = 1;
  vref(1) = -2;
  vref(2) = 3;
  vref(3) = -4;

  Morpheus::IO::read_matrix_market_file(v, filename);

  CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(vref, v));
}

TEST(MatrixMarket, ReadArrayUnsymmetricRealDenseMatrix) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A, Aref(2, 2);
  std::string filename(get_mm_test_path() + "/array_unsymmetric_real.mtx");

  Aref(0, 0) = 1.11;
  Aref(0, 1) = -2.22;
  Aref(1, 0) = 3.33;
  Aref(1, 1) = -4.44;

  Morpheus::IO::read_matrix_market_file(A, filename);

  CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(Aref, A));
}

TEST(MatrixMarket, ReadArrayUnsymmetricRealDenseVector) {
  Morpheus::DenseVector<double, Morpheus::HostSpace> v, vref(4);
  std::string filename(get_mm_test_path() + "/vec_unsymmetric_real.mtx");

  vref(0) = 1.11;
  vref(1) = -2.22;
  vref(2) = 3.33;
  vref(3) = -4.44;

  Morpheus::IO::read_matrix_market_file(v, filename);

  CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(vref, v));
}

TEST(MatrixMarket, ReadArrayInvalidSize) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/array_invalid_size.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadArrayInvalidTypePattern) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/array_invalid_type_pattern.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, ReadArrayInvalidTypeComplex) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/array_invalid_type_complex.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, ReadArrayInvalidTypeCustom) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/array_invalid_type_custom.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadArrayUnexpectedEOF) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() + "/array_unexpected_eof.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::IOException);
}

TEST(MatrixMarket, ReadArrayInvalidSymmetrySymmetric) {
  Morpheus::DenseMatrix<double, Morpheus::HostSpace> A;
  std::string filename(get_mm_test_path() +
                       "/array_invalid_symmetry_symmetric.mtx");

  EXPECT_THROW(Morpheus::IO::read_matrix_market_file(A, filename),
               Morpheus::NotImplementedException);
}

TEST(MatrixMarket, WriteMatrixDoubleGeneralCoo) {
  Morpheus::CooMatrix<double, Morpheus::HostSpace> A(2, 2, 4), Aref;
  // create_mm_out_path();
  std::string filename(get_mm_out_path() + "/coo_double.mtx");

  A.row_indices(0)    = 0;
  A.column_indices(0) = 0;
  A.values(0)         = 1.11;
  A.row_indices(1)    = 0;
  A.column_indices(1) = 1;
  A.values(1)         = 2.22;
  A.row_indices(2)    = 1;
  A.column_indices(2) = 0;
  A.values(2)         = 3.33;
  A.row_indices(3)    = 1;
  A.column_indices(3) = 1;
  A.values(3)         = 4.44;

  Morpheus::IO::write_matrix_market_file(A, filename);

  Morpheus::IO::read_matrix_market_file(Aref, filename);
  CHECK_COO_CONTAINERS(A, Aref);
  EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
}

// TEST(MatrixMarket, WriteMatrixFloatGeneralCoo) {
//   Morpheus::CooMatrix<float, Morpheus::HostSpace> A(2, 2, 4), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/coo_float.mtx");

//   A.row_indices(0)    = 0;
//   A.column_indices(0) = 0;
//   A.values(0)         = 1.11;
//   A.row_indices(1)    = 0;
//   A.column_indices(1) = 1;
//   A.values(1)         = 2.22;
//   A.row_indices(2)    = 1;
//   A.column_indices(2) = 0;
//   A.values(2)         = 3.33;
//   A.row_indices(3)    = 1;
//   A.column_indices(3) = 1;
//   A.values(3)         = 4.44;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_COO_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteMatrixIntGeneralCoo) {
//   Morpheus::CooMatrix<int, Morpheus::HostSpace> A(2, 2, 4), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/coo_int.mtx");

//   A.row_indices(0)    = 0;
//   A.column_indices(0) = 0;
//   A.values(0)         = 1;
//   A.row_indices(1)    = 0;
//   A.column_indices(1) = 1;
//   A.values(1)         = 2;
//   A.row_indices(2)    = 1;
//   A.column_indices(2) = 0;
//   A.values(2)         = 3;
//   A.row_indices(3)    = 1;
//   A.column_indices(3) = 1;
//   A.values(3)         = 4;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_COO_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteMatrixLongGeneralCoo) {
//   Morpheus::CooMatrix<long long, Morpheus::HostSpace> A(2, 2, 4), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/coo_long.mtx");

//   A.row_indices(0)    = 0;
//   A.column_indices(0) = 0;
//   A.values(0)         = 1;
//   A.row_indices(1)    = 0;
//   A.column_indices(1) = 1;
//   A.values(1)         = 2;
//   A.row_indices(2)    = 1;
//   A.column_indices(2) = 0;
//   A.values(2)         = 3;
//   A.row_indices(3)    = 1;
//   A.column_indices(3) = 1;
//   A.values(3)         = 4;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_COO_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteMatrixDynamic) {
//   Morpheus::DynamicMatrix<double, Morpheus::HostSpace> A, Aref;
//   Morpheus::CooMatrix<double, Morpheus::HostSpace> Acoo(2, 2, 4);
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/dynamic.mtx");

//   Acoo.row_indices(0)    = 0;
//   Acoo.column_indices(0) = 0;
//   Acoo.values(0)         = 1.11;
//   Acoo.row_indices(1)    = 0;
//   Acoo.column_indices(1) = 1;
//   Acoo.values(1)         = 2.22;
//   Acoo.row_indices(2)    = 1;
//   Acoo.column_indices(2) = 0;
//   Acoo.values(2)         = 3.33;
//   Acoo.row_indices(3)    = 1;
//   Acoo.column_indices(3) = 1;
//   Acoo.values(3)         = 4.44;

//   A = Acoo;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_DYNAMIC_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteMatrixDynamicWrongState) {
//   Morpheus::DynamicMatrix<double, Morpheus::HostSpace> A;
//   Morpheus::CooMatrix<double, Morpheus::HostSpace> Acoo(2, 2, 4);
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/dynamic.mtx");

//   Acoo.row_indices(0)    = 0;
//   Acoo.column_indices(0) = 0;
//   Acoo.values(0)         = 1.11;
//   Acoo.row_indices(1)    = 0;
//   Acoo.column_indices(1) = 1;
//   Acoo.values(1)         = 2.22;
//   Acoo.row_indices(2)    = 1;
//   Acoo.column_indices(2) = 0;
//   Acoo.values(2)         = 3.33;
//   Acoo.row_indices(3)    = 1;
//   Acoo.column_indices(3) = 1;
//   Acoo.values(3)         = 4.44;

//   A = Acoo;
//   Morpheus::convert<Morpheus::Serial>(A, Morpheus::CSR_FORMAT);
//   EXPECT_THROW(Morpheus::IO::write_matrix_market_file(A, filename),
//                Morpheus::RuntimeException);
// }

// TEST(MatrixMarket, WriteArrayDoubleGeneralVec) {
//   Morpheus::DenseVector<double, Morpheus::HostSpace> v(4), vref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/vec_double.mtx");

//   v(0) = 1.11;
//   v(1) = -2.22;
//   v(2) = 3.33;
//   v(3) = -4.44;

//   Morpheus::IO::write_matrix_market_file(v, filename);

//   Morpheus::IO::read_matrix_market_file(vref, filename);
//   CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(v, vref));
// }

// TEST(MatrixMarket, WriteArrayFloatGeneralVec) {
//   Morpheus::DenseVector<float, Morpheus::HostSpace> v(4), vref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/vec_float.mtx");

//   v(0) = 1.11;
//   v(1) = -2.22;
//   v(2) = 3.33;
//   v(3) = -4.44;

//   Morpheus::IO::write_matrix_market_file(v, filename);

//   Morpheus::IO::read_matrix_market_file(vref, filename);
//   CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(v, vref));
// }

// TEST(MatrixMarket, WriteArrayIntGeneralVec) {
//   Morpheus::DenseVector<int, Morpheus::HostSpace> v(4), vref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/vec_int.mtx");

//   v(0) = 1;
//   v(1) = -2;
//   v(2) = 3;
//   v(3) = -4;

//   Morpheus::IO::write_matrix_market_file(v, filename);

//   Morpheus::IO::read_matrix_market_file(vref, filename);
//   CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(v, vref));
// }

// TEST(MatrixMarket, WriteArrayLongGeneralVec) {
//   Morpheus::DenseVector<long long, Morpheus::HostSpace> v(4), vref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/vec_long.mtx");

//   v(0) = 1;
//   v(1) = -2;
//   v(2) = 3;
//   v(3) = -4;

//   Morpheus::IO::write_matrix_market_file(v, filename);

//   Morpheus::IO::read_matrix_market_file(vref, filename);
//   CHECK_DENSE_VECTOR_CONTAINERS(v, vref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(v, vref));
// }

// TEST(MatrixMarket, WriteArrayDoubleGeneralMat) {
//   Morpheus::DenseMatrix<double, Morpheus::HostSpace> A(2, 2), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/mat_double.mtx");

//   A(0, 0) = 1.11;
//   A(0, 1) = -2.22;
//   A(1, 0) = 3.33;
//   A(1, 1) = -4.44;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteArrayFloatGeneralMat) {
//   Morpheus::DenseMatrix<float, Morpheus::HostSpace> A(2, 2), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/mat_float.mtx");

//   A(0, 0) = 1.11;
//   A(0, 1) = -2.22;
//   A(1, 0) = 3.33;
//   A(1, 1) = -4.44;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteArrayIntGeneralMat) {
//   Morpheus::DenseMatrix<int, Morpheus::HostSpace> A(2, 2), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/mat_int.mtx");

//   A(0, 0) = 1;
//   A(0, 1) = -2;
//   A(1, 0) = 3;
//   A(1, 1) = -4;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

// TEST(MatrixMarket, WriteArrayLongGeneralMat) {
//   Morpheus::DenseMatrix<long long, Morpheus::HostSpace> A(2, 2), Aref;
//   create_mm_out_path();
//   std::string filename(get_mm_out_path() + "/mat_long.mtx");

//   A(0, 0) = 1;
//   A(0, 1) = -2;
//   A(1, 0) = 3;
//   A(1, 1) = -4;

//   Morpheus::IO::write_matrix_market_file(A, filename);

//   Morpheus::IO::read_matrix_market_file(Aref, filename);
//   CHECK_DENSE_MATRIX_CONTAINERS(A, Aref);
//   EXPECT_TRUE(Morpheus::Test::have_same_data(A, Aref));
// }

}  // namespace Test

#endif  // TEST_CORE_TEST_MATRIXMARKET_HPP
