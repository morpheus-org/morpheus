/**
 * Macros_CooMatrix.hpp
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

#ifndef TEST_CORE_UTILS_MACROS_DEFINITIONS_HPP
#define TEST_CORE_UTILS_MACROS_DEFINITIONS_HPP

const size_t SMALL_MATRIX_NROWS     = 10;
const size_t SMALL_MATRIX_NCOLS     = 10;
const size_t SMALL_MATRIX_NNZ       = 32;
const size_t SMALL_MATRIX_ALIGNMENT = 32;

const size_t SMALL_DIA_MATRIX_NDIAGS = 9;

const size_t SMALL_VECTOR_SIZE = 10;

// clang-format off
// Small Matrix
//        0        1        2        3      4        5     6        7       8     9
// 0 [ 01.11 |       |        | 02.22 |       |       |       | 03.33 | 04.44 |       ]
// 1 [       | 05.55 |        |       | 06.66 |       |       | 07.77 |       | 08.88 ]
// 2 [       |       | 09.99  |       |       | 10.10 |       |       |       |       ]
// 3 [ 11.11 |       |        | 12.12 |       |       | 13.13 |       |       |       ]
// 4 [       | 14.14 |        |       | 15.15 |       |       | 16.16 |       |       ]
// 5 [       |       | 17.17  |       |       | 18.18 |       |       | 19.19 |       ]
// 6 [       |       |        | 20.20 |       |       | 21.21 |       |       | 22.22 ]
// 7 [ 23.23 | 24.24 |        |       | 25.25 |       |       | 26.26 |       |       ]
// 8 [ 27.27 |       |        |       |       | 28.28 |       |       | 29.29 |       ]
// 9 [       | 30.30 |        |       |       |       | 31.31 |       |       | 32.32 ]

// Updated Small Matrix
//      0        1        2        3      4        5     6        7       8     9
// 0 [ 01.11 |        |        | 02.22 |        |       |       | 03.33 | -04.44 |        ]
// 1 [       |  05.55 |        |       |  06.66 |       |       | 07.77 |        | -08.88 ]
// 2 [       |        | 09.99  |       |        | 10.10 |       |       |        |        ]
// 3 [ 11.11 |        |        | 12.12 |        |       | 13.13 |       |        |        ]
// 4 [       | -14.14 |        |       | -15.15 |       |       | 16.16 |        |        ]
// 5 [       |        | 17.17  |       |        | 18.18 |       |       |  19.19 |        ]
// 6 [       |        |        | 20.20 |        |       | 21.21 |       |        | 22.22  ]
// 7 [ 23.23 |  24.24 |        |       | -25.25 |       |       | 26.26 |        |        ]
// 8 [ 27.27 |        |        |       |        | 28.28 |       |       |  29.29 |        ]
// 9 [       |  30.30 |        |       |        |       | 31.31 |       |        | 32.32  ]
// clang-format on

#endif  // TEST_CORE_UTILS_MACROS_DEFINITIONS_HPP
