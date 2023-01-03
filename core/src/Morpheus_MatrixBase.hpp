/**
 * Morpheus_MatrixBase.hpp
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
#ifndef MORPHEUS_MATRIXBASE_HPP
#define MORPHEUS_MATRIXBASE_HPP

#include <Morpheus_MatrixOptions.hpp>
#include <Morpheus_ContainerTraits.hpp>

namespace Morpheus {

/**
 * \addtogroup base_containers Base Containers
 * \brief Containers used as base to derive others
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Base class used to derive new matrices.
 *
 * @tparam Container Type of the new container we are deriving.
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The MatrixBase class is used to organize common information that is often
 * found across the different matrix types/formats. Examples of such information
 * is the shape of the matrix, a specific structure might have (e.g Symmetric)
 * or any specific properties such as it has short rows.
 *
 * \par Example
 * The example below shows how to define a new matrix class that will inherit
 * from MatrixBase.
 * \code #include <Morpheus_Core.hpp>
 *
 * template <class ValueType, class... Properties>
 * class NewMatrix : public MatrixBase<NewMatrix, ValueType, Properties...>{
 *  using base = MatrixBase<NewMatrix, ValueType, Properties...>;
 *  // Implementation
 * }
 * \endcode
 *
 */
template <template <class, class...> class Container, class ValueType,
          class... Properties>
class MatrixBase : public ContainerTraits<Container, ValueType, Properties...> {
 public:
  //!< The complete type of the container
  using type = MatrixBase<Container, ValueType, Properties...>;
  //!< The traits associated with the particular container
  using traits = ContainerTraits<Container, ValueType, Properties...>;
  //!< The type of the indices held by the container
  using size_type = typename traits::size_type;
  /**
   * @brief Default constructor
   *
   */
  MatrixBase()
      : _m(0), _n(0), _nnz(0), _structure(MATSTR_NONE), _options(MATOPT_NONE) {}

  /**
   * @brief Construct a MatrixBase object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   */
  MatrixBase(size_type rows, size_type cols, size_type entries = 0)
      : _m(rows),
        _n(cols),
        _nnz(entries),
        _structure(MATSTR_NONE),
        _options(MATOPT_NONE) {}

  /**
   * @brief Resizes MatrixBase with shape of (num_rows, num_cols) and sets
   * number of non-zero entries to num_entries.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_entries Number of non-zero entries in resized matrix.
   */
  void resize(size_type rows, size_type cols, size_type entries) {
    _m   = rows;
    _n   = cols;
    _nnz = entries;
  }

  /**
   * @brief Number of rows of the matrix
   *
   * @return size_type
   */
  inline size_type nrows() const { return _m; }

  /**
   * @brief Number of columns of the matrix
   *
   * @return size_type
   */
  inline size_type ncols() const { return _n; }

  /**
   * @brief Number of non-zeros of the matrix
   *
   * @return size_type
   */
  inline size_type nnnz() const { return _nnz; }

  /**
   * @brief Set the number of rows of the matrix
   *
   * @param rows Number of rows
   */
  inline void set_nrows(const size_type rows) { _m = rows; }

  /**
   * @brief Set the number of columns of the matrix
   *
   * @param rows Number of columns
   */
  inline void set_ncols(const size_type cols) { _n = cols; }

  /**
   * @brief Set the number of non-zeros of the matrix
   *
   * @param rows Number of non-zeros
   */
  inline void set_nnnz(const size_type nnz) { _nnz = nnz; }

  /**
   * @brief The specialized structure of the matrix e.g Symmetric
   *
   * @return MatrixStructure
   */
  inline MatrixStructure structure() const { return _structure; }
  /**
   * @brief Information about specific characteristics of the matrix e.g has
   * short rows
   *
   * @return MatrixOptions
   */
  inline MatrixOptions options() const { return _options; }

  /**
   * @brief Set the structure of the matrix
   *
   * @param op Enum for the matrix structure
   */
  inline void set_structure(MatrixStructure op) { _structure = op; }

  /**
   * @brief Set the characteristics of the matrix
   *
   * @param op Enum for available options
   */
  inline void set_options(MatrixOptions op) { _options = op; }

 private:
  size_type _m, _n, _nnz;
  MatrixStructure _structure;
  MatrixOptions _options;
};
/*! \}  // end of base_containers group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_MATRIXBASE_HPP