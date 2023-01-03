/**
 * Morpheus_MatrixOperations.hpp
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

#ifndef MORPHEUS_MATRIXOPTIONS_HPP
#define MORPHEUS_MATRIXOPTIONS_HPP

namespace Morpheus {

typedef enum {
  MATSTR_NONE                   = 0,
  MATSTR_SYMMETRIC              = 1,
  MATSTR_STRUCTURALLY_SYMMETRIC = 2,
  MATSTR_HERMITIAN              = 3,
  MATSTR_SPD                    = 4,
} MatrixStructure;

typedef enum {
  MATOPT_NONE       = 0,
  MATOPT_SHORT_ROWS = 1,
} MatrixOptions;

}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXOPTIONS_HPP