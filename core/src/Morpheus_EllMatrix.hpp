/**
 * Morpheus_EllMatrix.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 - 2022 The University of Edinburgh
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

#ifndef MORPHEUS_ELLMATRIX_HPP
#define MORPHEUS_ELLMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_MatrixBase.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \brief Two-dimensional Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the ELLPACK (ELL) Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The EllMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container is the implementation of the ELLPACK Format.
 * It is a polymorphic container in the sense that it can store scalar or
 * integer type values, on host or device depending how the template parameters
 * are selected.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 * // Matrix to Build
 * // [1 * 2]
 * // [* * 3]
 * // [* 4 *]
 * int main(){
 *    TODO
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class EllMatrix : public MatrixBase<EllMatrix, ValueType, Properties...> {
 public:
  //!< The traits associated with the particular container
  using traits = ContainerTraits<EllMatrix, ValueType, Properties...>;
  //!< The complete type of the container
  using type = typename traits::type;
  using base = MatrixBase<EllMatrix, ValueType, Properties...>;
  //!< The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::EllFormatTag>::tag;

  /*! The type of the values held by the container - can be const */
  using value_type = typename traits::value_type;
  /*! The non-constant type of the values held by the container */
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::size_type;
  //!< The type of the indices held by the container - can be const
  using index_type = typename traits::index_type;
  /*! The non-constant type of the indices held by the container */
  using non_const_index_type = typename traits::non_const_index_type;

  using array_layout    = typename traits::array_layout;
  using backend         = typename traits::backend;
  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;
  using memory_traits   = typename traits::memory_traits;
  using HostMirror      = typename traits::HostMirror;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  // /*! The type of \p DenseVector that holds the index_type data */
  // using index_array_type =
  //     Morpheus::DenseVector<index_type, size_type, array_layout, backend,
  //                           memory_traits>;
  // using index_array_pointer = typename index_array_type::value_array_pointer;
  // using index_array_reference =
  //     typename index_array_type::value_array_reference;

  // /*! The type of \p DenseVector that holds the value_type data */
  // using value_array_type =
  //     Morpheus::DenseVector<value_type, size_type, array_layout, backend,
  //                           memory_traits>;
  // using value_array_pointer = typename value_array_type::value_array_pointer;
  // using value_array_reference =
  //     typename value_array_type::value_array_reference;

  /**
   * @brief The default destructor.
   */
  ~EllMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a EllMatrix container
   * from another EllMatrix container with the same properties.
   */
  EllMatrix(const EllMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a EllMatrix container
   * from another EllMatrix container with the same properties.
   */
  EllMatrix(EllMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a EllMatrix container
   * from another EllMatrix container with the same properties.
   */
  EllMatrix &operator=(const EllMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a EllMatrix container
   * from another EllMatrix container with the same properties.
   */
  EllMatrix &operator=(EllMatrix &&) = default;

  /**
   * @brief Construct an empty EllMatrix object
   */
  inline EllMatrix() : base() {
    throw Morpheus::NotImplementedException("TODO");
  }

  /**
   * @brief Construct a EllMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   */
  inline EllMatrix(const size_type num_rows, const size_type num_cols,
                   const size_type num_entries)
      : base(num_rows, num_cols, num_entries) {
    throw Morpheus::NotImplementedException("TODO");
  }

  // Construct from pointers
  // Construct from vectors/matrices

  /**
   * @brief Constructs a EllMatrix from another compatible EllMatrix
   *
   * @par Constructs a EllMatrix from another compatible EllMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  EllMatrix(const EllMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                EllMatrix, EllMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    throw Morpheus::NotImplementedException("TODO");
  }

  /**
   * @brief Assigns a EllMatrix from another compatible EllMatrix
   *
   * @par Overview
   * Assigns a EllMatrix from another compatible EllMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<EllMatrix, EllMatrix<VR, PR...>>::value,
      EllMatrix &>::type
  operator=(const EllMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    throw Morpheus::NotImplementedException("TODO");

    return *this;
  }

  // /**
  //  * @brief Constructs a EllMatrix from a compatible DynamicMatrix
  //  *
  //  * @par Overview
  //  * Constructs a EllMatrix from a compatible DynamicMatrix i.e a matrix that
  //  * satisfies the \p is_dynamically_compatible check. Note that when the
  //  active
  //  * type of the dynamic matrix is different from the concrete type, this
  //  will
  //  * result in an exception thrown.
  //  *
  //  * @tparam VR Type of Values the Other Matrix holds.
  //  * @tparam PR Properties of the Other Matrix.
  //  * @param src The matrix we are constructing from.
  //  */
  // template <class VR, class... PR>
  // EllMatrix(const DynamicMatrix<VR, PR...> &src,
  //           typename std::enable_if<is_dynamically_compatible<
  //               EllMatrix, DynamicMatrix<VR, PR...>>::value>::type * =
  //               nullptr)
  //     : base(src.nrows(), src.ncols(), src.nnnz()) {
  //   auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
  //                      std::ref(*this));

  //   std::visit(f, src.const_formats());
  // }

  // /**
  //  * @brief Assigns a EllMatrix from a compatible DynamicMatrix
  //  *
  //  * @par Overview
  //  * Assigns a EllMatrix from a compatible DynamicMatrix i.e a matrix that
  //  * satisfies the \p is_dynamically_compatible check. Note that when the
  //  active
  //  * type of the dynamic matrix is different from the concrete type, this
  //  will
  //  * result in an exception thrown.
  //  *
  //  * @tparam VR Type of Values the Other Matrix holds.
  //  * @tparam PR Properties of the Other Matrix.
  //  * @param src The matrix we are assigning from.
  //  */
  // template <class VR, class... PR>
  // typename std::enable_if<
  //     is_dynamically_compatible<EllMatrix, DynamicMatrix<VR, PR...>>::value,
  //     EllMatrix &>::type
  // operator=(const DynamicMatrix<VR, PR...> &src) {
  //   auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
  //                      std::ref(*this));

  //   std::visit(f, src.const_formats());

  //   return *this;
  // }

  /**
   * @brief Construct a EllMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  EllMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assign to EllMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes EllMatrix with shape of (num_rows, num_cols) and sets number
   * of non-zero entries to num_entries.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_entries Number of non-zero entries in resized matrix.
   */
  inline void resize(const size_type num_rows, const size_type num_cols,
                     const size_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    throw Morpheus::NotImplementedException("TODO");
  }

  /**
   * @brief Resizes EllMatrix with the shape and number of non-zero entries of
   * another EllMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source EllMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const EllMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
  }

  /**
   * @brief Allocates memory from another EllMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p EllMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline EllMatrix &allocate(const EllMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }

  /**
   * @brief Returns the format enum assigned to the EllMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * EllMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

 private:
  static constexpr formats_e _id = Morpheus::ELL_FORMAT;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_ELLMATRIX_HPP