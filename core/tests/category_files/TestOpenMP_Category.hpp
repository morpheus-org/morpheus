/**
 * TestOpenMP_Category.hpp
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

#ifndef MORPHEUS_CORE_TEST_OPENMP_HPP
#define MORPHEUS_CORE_TEST_OPENMP_HPP

#include <gtest/gtest.h>

#define TEST_CATEGORY openmp
#define TEST_CATEGORY_NUMBER 2
#define TEST_SPACE OpenMP
#define TEST_EXECSPACE Kokkos::OpenMP
#define TEST_CUSTOM_SPACE Morpheus::OpenMP
#define TEST_GENERIC_SPACE Morpheus::Generic::OpenMP
#define TEST_CATEGORY_FIXTURE(name) openmp_##name

#endif  // MORPHEUS_CORE_TEST_OPENMP_HPP