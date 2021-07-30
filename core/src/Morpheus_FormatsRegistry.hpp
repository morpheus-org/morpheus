/**
 * Morpheus_FormatsRegistry.hpp
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

#ifndef MORPHEUS_FORMATSREGISTRY_HPP
#define MORPHEUS_FORMATSREGISTRY_HPP

#include <Morpheus_CooMatrix.hpp>
#include <Morpheus_CsrMatrix.hpp>
#include <Morpheus_DiaMatrix.hpp>

#include <impl/Morpheus_MatrixProxy.hpp>

namespace Morpheus {

template <class ValueType, class... Properties>
struct MatrixFormats {
  using formats_proxy =
      typename MatrixFormatsProxy<CooMatrix<ValueType, Properties...>,
                                  CsrMatrix<ValueType, Properties...>,
                                  DiaMatrix<ValueType, Properties...>>::type;
  using variant   = typename formats_proxy::variant;
  using type_list = typename formats_proxy::type_list;
};

// Enums should be in the same order as types in MatrixFormatsProxy
enum formats_e { COO_FORMAT = 0, CSR_FORMAT, DIA_FORMAT };

}  // namespace Morpheus

#endif  // MORPHEUS_FORMATSREGISTRY_HPP