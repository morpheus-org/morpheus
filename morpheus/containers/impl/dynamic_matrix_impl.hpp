/**
 * dynamic_matrix_impl.hpp
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

#ifndef MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_IMPL_HPP
#define MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_IMPL_HPP

#include <string>
#include <morpheus/core/exceptions.hpp>
#include <morpheus/core/matrix_proxy.hpp>
#include <morpheus/core/matrix_traits.hpp>

#include <morpheus/containers/coo_matrix.hpp>
#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/dia_matrix.hpp>

namespace Morpheus {

template <class... Properties>
struct MatrixFormats {
  using formats_proxy =
      typename MatrixFormatsProxy<CooMatrix<Properties...>,
                                  CsrMatrix<Properties...>,
                                  DiaMatrix<Properties...>>::type;
  using variant   = typename formats_proxy::variant;
  using type_list = typename formats_proxy::type_list;
};
// Enums should be in the same order as types in MatrixFormatsProxy
enum formats_e { COO_FORMAT = 0, CSR_FORMAT, DIA_FORMAT };

namespace Impl {

template <class... Properties>
struct any_type_resize : public Impl::MatrixTraits<Properties...> {
  using traits      = Impl::MatrixTraits<Properties...>;
  using index_type  = typename traits::index_type;
  using value_type  = typename traits::value_type;
  using result_type = void;

  // Specialization for Coo resize with three arguments
  template <typename... Args>
  result_type operator()(CooMatrix<Properties...> &mat, const index_type nrows,
                         const index_type ncols, const index_type nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Csr resize with three arguments
  template <typename... Args>
  result_type operator()(CsrMatrix<Properties...> &mat, const index_type nrows,
                         const index_type ncols, const index_type nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Dia resize with four arguments
  template <typename... Args>
  result_type operator()(DiaMatrix<Properties...> &mat, const index_type nrows,
                         const index_type ncols, const index_type nnnz,
                         const index_type ndiag) {
    mat.resize(nrows, ncols, nnnz, ndiag);
  }

  // Specialization for Dia resize with five arguments
  template <typename... Args>
  result_type operator()(DiaMatrix<Properties...> &mat, const index_type nrows,
                         const index_type ncols, const index_type nnnz,
                         const index_type ndiag, const index_type alignment) {
    mat.resize(nrows, ncols, nnnz, ndiag, alignment);
  }
  // Specialization for any other case and dummy overlads
  // eg Resize with four arguments is not supported by Coo
  //    though needed for compiling dynamic matrix interface
  template <typename T, typename... Args>
  result_type operator()(T &mat, Args &&...args) {
    std::string str_args = Morpheus::append_str(args...);
    throw std::runtime_error(
        "Invalid use of the dynamic resize interface.\n\
                mat.resize(" +
        str_args + ") for " + mat.name() + " format.");
  }
};

struct any_type_get_name {
  using result_type = std::string;
  template <typename T>
  result_type operator()(const T &mat) const {
    return mat.name();
  }
};

struct any_type_get_nrows {
  template <typename T>
  typename T::index_type operator()(const T &mat) const {
    return mat.nrows();
  }
};

struct any_type_get_ncols {
  template <typename T>
  typename T::index_type operator()(const T &mat) const {
    return mat.ncols();
  }
};

struct any_type_get_nnnz {
  template <typename T>
  typename T::index_type operator()(const T &mat) const {
    return mat.nnnz();
  }
};

template <size_t I, typename... Properties>
struct activate_impl {
  using variant   = typename MatrixFormats<Properties...>::variant;
  using type_list = typename MatrixFormats<Properties...>::type_list;

  static void activate(variant &A, size_t idx) {
    if (idx == I - 1) {
      A = typename type_list::template type<I - 1>{};
    } else {
      activate_impl<I - 1, Properties...>::activate(A, idx);
    }
  }
};

// Base case, activate to the first type in the variant
template <typename... Properties>
struct activate_impl<0, Properties...> {
  using variant   = typename MatrixFormats<Properties...>::variant;
  using type_list = typename MatrixFormats<Properties...>::type_list;

  static void activate(variant &A, size_t idx) {
    idx = 0;
    activate_impl<1, Properties...>::activate(A, idx);
  }
};

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_IMPL_HPP