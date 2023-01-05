/**
 * Morpheus_MatrixAnalytics_Impl.hpp
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

#ifndef MORPHEUS_IMPL_MATRIXANALYTICS_IMPL_HPP
#define MORPHEUS_IMPL_MATRIXANALYTICS_IMPL_HPP

// TODO: Let Cmake autogenerate those
#include <impl/Coo/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Coo/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Coo/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Coo/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Coo/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Coo/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Coo/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Coo/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#include <impl/Csr/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Csr/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Csr/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Csr/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Csr/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Csr/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#include <impl/Dia/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Dia/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Dia/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Dia/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Dia/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Dia/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Dia/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Dia/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Dia/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Dia/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#include <impl/Ell/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Ell/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Ell/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Ell/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Ell/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Ell/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Ell/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Ell/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Ell/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Ell/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#include <impl/Hyb/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hyb/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hyb/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hyb/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hyb/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hyb/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hyb/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hyb/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hyb/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hyb/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#include <impl/Hdc/Serial/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hdc/OpenMP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hdc/Cuda/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hdc/HIP/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hdc/Kokkos/Morpheus_NonZeros_Per_Row_Impl.hpp>
#include <impl/Hdc/Serial/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hdc/OpenMP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hdc/Cuda/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hdc/HIP/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>
#include <impl/Hdc/Kokkos/Morpheus_NonZeros_Per_Diagonal_Impl.hpp>

#endif  // MORPHEUS_IMPL_MATRIXANALYTICS_IMPL_HPP