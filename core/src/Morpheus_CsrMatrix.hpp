/**
 * Morpheus_CsrMatrix.hpp
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

#ifndef MORPHEUS_CSRMATRIX_HPP
#define MORPHEUS_CSRMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DynamicMatrix.hpp>
#include <Morpheus_MatrixBase.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the Compressed-Sparse Row (CSR) Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The CsrMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container is the compressed implementation of the COO
 * Format, where the row indices array is compressed. In general, it allows for
 * fast row access and matrix-vector multiplications and it is one of the most
 * widely used formats. It is a polymorphic container in the sense that it can
 * store scalar or integer type values, on host or device depending how the
 * template parameters are selected.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 * // Matrix to Build
 * // [1 * 2]
 * // [* * 3]
 * // [* 4 *]
 * int main(){
 *  using Matrix = Morpheus::CsrMatrix<double, Kokkos::HostSpace>;
 *  using index_array_type = typename Matrix::index_array_type;
 *  using value_array_type = typename Matrix::value_array_type;
 *
 *  index_array_type off(4, 0), j(4, 0);
 *  value_array_type v(4,0);
 *
 *  off[0] = 0; off[1] = 2; off[2] = 3; off[3] = 4;
 *
 *  j[0] = 0; v[0] = 1;
 *  j[1] = 2; v[1] = 2;
 *  j[2] = 2; v[2] = 3;
 *  j[3] = 1; v[3] = 4;
 *
 *  // Construct the matrix from off,j,v
 *  Matrix A(3, 3, 4, off, j, v);
 *
 *  Morpheus::print(A); // prints A
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class CsrMatrix : public MatrixBase<CsrMatrix, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = ContainerTraits<CsrMatrix, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  using base = MatrixBase<CsrMatrix, ValueType, Properties...>;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::CsrFormatTag>::tag;

  /*! The type of the values held by the container - can be const */
  using value_type = typename traits::value_type;
  /*! The non-constant type of the values held by the container */
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::size_type;
  /*! The type of the indices held by the container - can be const */
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

  /*! The type of \p DenseVector that holds the index_type data */
  using index_array_type =
      Morpheus::DenseVector<index_type, size_type, array_layout, backend,
                            memory_traits>;
  using index_array_pointer = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;

  /*! The type of \p DenseVector that holds the value_type data */
  using value_array_type =
      Morpheus::DenseVector<value_type, size_type, array_layout, backend,
                            memory_traits>;
  using value_array_pointer = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;

  /**
   * @brief The default destructor.
   */
  ~CsrMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a CsrMatrix container
   * from another CsrMatrix container with the same properties.
   */
  CsrMatrix(const CsrMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a CsrMatrix container
   * from another CsrMatrix container with the same properties.
   */
  CsrMatrix(CsrMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a CsrMatrix container
   * from another CsrMatrix container with the same properties.
   */
  CsrMatrix &operator=(const CsrMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a CsrMatrix container
   * from another CsrMatrix container with the same properties.
   */
  CsrMatrix &operator=(CsrMatrix &&) = default;

  /**
   * @brief Construct an empty CsrMatrix object
   */
  inline CsrMatrix() : base(), _row_offsets(), _column_indices(), _values() {}

  /**
   * @brief Construct a CsrMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   */
  inline CsrMatrix(const size_type num_rows, const size_type num_cols,
                   const size_type num_entries)
      : base(num_rows, num_cols, num_entries),
        _row_offsets((num_rows + (size_type)1)),
        _column_indices(num_entries),
        _values(num_entries) {}

  // Construct from raw pointers
  template <typename ValuePtr, typename IndexPtr>
  explicit inline CsrMatrix(
      const size_type num_rows, const size_type num_cols,
      const size_type num_entries, IndexPtr roff_ptr, IndexPtr cind_ptr,
      ValuePtr vals_ptr,
      typename std::enable_if<
          (std::is_pointer<ValuePtr>::value &&
           is_same_value_type<value_type, ValuePtr>::value &&
           memory_traits::is_unmanaged) &&
          (std::is_pointer<IndexPtr>::value &&
           is_same_index_type<index_type, IndexPtr>::value &&
           memory_traits::is_unmanaged)>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_offsets(num_rows + 1, roff_ptr),
        _column_indices(num_entries, cind_ptr),
        _values(num_entries, vals_ptr) {}

  /**
   * @brief Construct a CsrMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries and assign the indices and values
   * from \p DenseVector arrays.
   *
   * @tparam ValueArray Value type DenseVector type.
   * @tparam IndexArray Index type DenseVector type.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   * @param rind \p DenseVector containing the row indices of the matrix.
   * @param cind \p DenseVector containing the column indices of the matrix.
   * @param vals \p DenseVector containing the values of the matrix.
   */
  template <typename ValueArray, typename IndexArray>
  explicit inline CsrMatrix(
      const size_type num_rows, const size_type num_cols,
      const size_type num_entries, IndexArray roff, IndexArray cind,
      ValueArray vals,
      typename std::enable_if<
          is_dense_vector_format_container<ValueArray>::value &&
          is_dense_vector_format_container<IndexArray>::value &&
          is_compatible<typename CsrMatrix::value_array_type,
                        ValueArray>::value &&
          is_compatible<typename CsrMatrix::index_array_type,
                        IndexArray>::value &&
          !ValueArray::memory_traits::is_unmanaged &&
          !IndexArray::memory_traits::is_unmanaged>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_offsets(roff),
        _column_indices(cind),
        _values(vals) {}

  /**
   * @brief Constructs a CsrMatrix from another compatible CsrMatrix
   *
   * @par Constructs a CsrMatrix from another compatible CsrMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  CsrMatrix(const CsrMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                CsrMatrix, CsrMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _row_offsets(src.crow_offsets()),
        _column_indices(src.ccolumn_indices()),
        _values(src.cvalues()) {}

  /**
   * @brief Assigns a CsrMatrix from another compatible CsrMatrix
   *
   * @par Overview
   * Assigns a CsrMatrix from another compatible CsrMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<CsrMatrix, CsrMatrix<VR, PR...>>::value,
      CsrMatrix &>::type
  operator=(const CsrMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _row_offsets    = src.crow_offsets();
    _column_indices = src.ccolumn_indices();
    _values         = src.cvalues();

    return *this;
  }

  /**
   * @brief Constructs a CsrMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a CsrMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  CsrMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_dynamically_compatible<
                CsrMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  /**
   * @brief Assigns a CsrMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a CsrMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_dynamically_compatible<CsrMatrix, DynamicMatrix<VR, PR...>>::value,
      CsrMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  /**
   * @brief Construct a CsrMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  CsrMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assigns to CsrMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes CsrMatrix with shape of (num_rows, num_cols) and sets number
   * of non-zero entries to num_entries.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_entries Number of non-zero entries in resized matrix.
   */
  inline void resize(const size_type num_rows, const size_type num_cols,
                     const size_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    _row_offsets.resize(num_rows + 1);
    _column_indices.resize(num_entries);
    _values.resize(num_entries);
  }

  /**
   * @brief Resizes CsrMatrix with the shape and number of non-zero entries of
   * another CsrMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source CsrMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const CsrMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
  }

  /**
   * @brief Allocates memory from another CsrMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p CsrMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline CsrMatrix &allocate(const CsrMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }

  /**
   * @brief Returns the format enum assigned to the CsrMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * CsrMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

  /**
   * @brief Returns a reference to the row offset of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Row offset at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference row_offsets(size_type n) {
    return _row_offsets(n);
  }

  /**
   * @brief Returns a reference to the column index of the matrix with index \p
   * n
   *
   * @param n Index of the value to extract
   * @return Column index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  column_indices(size_type n) {
    return _column_indices(n);
  }

  /**
   * @brief Returns a reference to the value of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(size_type n) {
    return _values(n);
  }

  /**
   * @brief Returns a const-reference to the row offset of the matrix with index
   * \p n
   *
   * @param n Index of the value to extract
   * @return Row offset at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_reference
  crow_offsets(size_type n) const {
    return _row_offsets(n);
  }

  /**
   * @brief Returns a const-reference to the column index of the matrix with
   * index \p n
   *
   * @param n Index of the value to extract
   * @return Column index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_reference
  ccolumn_indices(size_type n) const {
    return _column_indices(n);
  }

  /**
   * @brief Returns a const-reference to the value of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const value_array_reference
  cvalues(size_type n) const {
    return _values(n);
  }

  /**
   * @brief Returns a reference to the row offsets of the matrix.
   *
   * @return index_array_type&  A reference to the row offsets.
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &row_offsets() {
    return _row_offsets;
  }

  /**
   * @brief Returns a reference to the column indices of the matrix.
   *
   * @return index_array_type&  A reference to the column indices.
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &column_indices() {
    return _column_indices;
  }

  /**
   * @brief Returns a reference to the values of the matrix.
   *
   * @return value_array_type&  A reference to the values.
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_type &values() { return _values; }

  /**
   * @brief Returns a const-reference to the row offsets of the matrix.
   *
   * @return const index_array_type&  A const reference to the row offsets.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_type &crow_offsets() const {
    return _row_offsets;
  }

  /**
   * @brief Returns a const-reference to the column indices of the matrix.
   *
   * @return index_array_type&  A const-reference to the column indices.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_type &ccolumn_indices()
      const {
    return _column_indices;
  }

  /**
   * @brief Returns a reference to the values of the matrix.
   *
   * @return values_array_type&  A reference to the values.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const value_array_type &cvalues() const {
    return _values;
  }

 private:
  index_array_type _row_offsets, _column_indices;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::CSR_FORMAT;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_CSRMATRIX_HPP