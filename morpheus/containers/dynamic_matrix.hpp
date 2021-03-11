/**
 * dynamic_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP
#define MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP

#include <string>
#include <variant>

#include <morpheus/containers/coo_matrix.hpp>
#include <morpheus/containers/csr_matrix.hpp>
#include <morpheus/containers/dia_matrix.hpp>
#include <morpheus/core/matrix_traits.hpp>

namespace Morpheus {
namespace Impl {
template <typename IndexType, typename ValueType>
struct any_type_resize {
  using result_type = void;

  // Specialization for Coo resize with three arguments
  template <typename... Args>
  result_type operator()(CooMatrix<IndexType, ValueType> &mat, IndexType nrows,
                         IndexType ncols, IndexType nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Csr resize with three arguments
  template <typename... Args>
  result_type operator()(CsrMatrix<IndexType, ValueType> &mat, IndexType nrows,
                         IndexType ncols, IndexType nnnz) {
    mat.resize(nrows, ncols, nnnz);
  }

  // Specialization for Dia resize with four arguments
  template <typename... Args>
  result_type operator()(DiaMatrix<IndexType, ValueType> &mat, IndexType nrows,
                         IndexType ncols, IndexType nnnz, IndexType ndiag) {
    mat.resize(nrows, ncols, nnnz, ndiag);
  }

  // Specialization for Dia resize with five arguments
  template <typename... Args>
  result_type operator()(DiaMatrix<IndexType, ValueType> &mat, IndexType nrows,
                         IndexType ncols, IndexType nnnz, IndexType ndiag,
                         IndexType alignment) {
    mat.resize(nrows, ncols, nnnz, ndiag, alignment);
  }
  // Specialization for any other case and dummy overlads
  // eg Resize with four arguments is not supported by Coo
  //    though needed for compiling dynamic matrix interface
  template <typename T, typename... Args>
  result_type operator()(T &mat, Args &&...args) {
    // TODO: Find another way to disable the execution of those overloads
    //       preferably at compile time.

    throw std::runtime_error(
        "Error::Invalid use of the dynamic resize interface");
  }
};

struct any_type_get_name {
  using result_type = std::string;
  template <typename T>
  result_type operator()(T &mat) const {
    return mat.name();
  }
};

struct any_type_get_nrows {
  template <typename T>
  typename T::index_type operator()(T &mat) const {
    return mat.nrows();
  }
};

struct any_type_get_ncols {
  template <typename T>
  typename T::index_type operator()(T &mat) const {
    return mat.ncols();
  }
};

struct any_type_get_nnnz {
  template <typename T>
  typename T::index_type operator()(T &mat) const {
    return mat.nnnz();
  }
};
}  // namespace Impl

// example:
// DynamicMatrix<int, double, memspace, formatspace>;
// where formatspace =
// std::variant<CooMatrix<int,double>,CsrMatrix<int,double>...>
template <class... Properties>
class DynamicMatrix : public Impl::MatrixTraits<FormatType<Impl::DynamicFormat>,
                                                Properties...> {
 public:
  using type = DynamicMatrix<Properties...>;
  using traits =
      Impl::MatrixTraits<FormatType<Impl::DynamicFormat>, Properties...>;
  using index_type  = typename traits::index_type;
  using value_type  = typename traits::value_type;
  using format_type = typename traits::format_type;

  using reference       = DynamicMatrix &;
  using const_reference = const DynamicMatrix &;

 private:
  using MatrixFormats = std::variant<
      Morpheus::CooMatrix<int, double>, Morpheus::CsrMatrix<int, double>,
      Morpheus::DiaMatrix<int, double>, Morpheus::CooMatrix<int, double>>;

 public:
  inline DynamicMatrix() = default;

  template <typename Format>
  inline DynamicMatrix(const Format &mat) : _formats(mat) {}

  template <typename... Args>
  inline void resize(int m, int n, int nnz, Args &&...args) {
    return std::visit(std::bind(Impl::any_type_resize<index_type, value_type>(),
                                std::placeholders::_1, m, n, nnz,
                                std::forward<Args>(args)...),
                      _formats);
  }

  inline std::string name() { return _name; }

  inline index_type nrows() {
    return std::visit(Impl::any_type_get_nrows(), _formats);
  }

  inline index_type ncols() {
    return std::visit(Impl::any_type_get_ncols(), _formats);
  }

  inline index_type nnnz() {
    return std::visit(Impl::any_type_get_nnnz(), _formats);
  }

  inline std::string active_name() {
    return std::visit(Impl::any_type_get_name(), _formats);
  }

  inline int active_index() { return _formats.index(); }

  // inline MatrixFormats& formats(){ return _formats; }

 private:
  std::string _name = "DynamicMatrix";
  MatrixFormats _formats;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DYNAMIC_MATRIX_HPP