/**
 * Morpheus_CooMatrix.hpp
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

#ifndef MORPHEUS_COOMATRIX_HPP
#define MORPHEUS_COOMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DynamicMatrix.hpp>
#include <Morpheus_Sort.hpp>
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
 * @brief Implementation of the Coordinate (COO) Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The CooMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container is the implementation of the Coordinate
 * Format (COO), that is also known as the "triplet" format because it stores
 * pairs of "ijv" of the matrix. In general, it is a fast and convenient format
 * for constructing sparse matrices. It is a polymorphic container in the sense
 * that it can store scalar or integer type values, on host or device depending
 * how the template parameters are selected.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 * // Matrix to Build
 * // [1 * 2]
 * // [* * 3]
 * // [* 4 *]
 * int main(){
 *  using Matrix = Morpheus::CooMatrix<double, Kokkos::HostSpace>;
 *  using index_array_type = typename Matrix::index_array_type;
 *  using value_array_type = typename Matrix::value_array_type;
 *
 *  index_array_type i(4, 0), j(4, 0);
 *  value_array_type v(4,0);
 *
 *  i[0] = 0; j[0] = 0; v[0] = 1;
 *  i[1] = 0; j[1] = 2; v[1] = 2;
 *  i[2] = 1; j[2] = 2; v[2] = 3;
 *  i[3] = 2; j[3] = 1; v[3] = 4;
 *
 *  // Construct the matrix from i,j,v
 *  Matrix A(3, 3, 4, i, j, v);
 *
 *  Morpheus::print(A); // prints A
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class CooMatrix : public MatrixBase<CooMatrix, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = ContainerTraits<CooMatrix, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  using base = MatrixBase<CooMatrix, ValueType, Properties...>;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::CooFormatTag>::tag;

  /*! The type of the values held by the container - can be const */
  using value_type = typename traits::value_type;
  /*! The non-constant type of the values held by the container */
  using non_const_value_type = typename traits::non_const_value_type;
  using size_type            = typename traits::index_type;
  /*! The type of the indices held by the container - can be const */
  using index_type = typename traits::index_type;
  /*! The non-constant type of the indices held by the container */
  using non_const_index_type = typename traits::non_const_index_type;

  using array_layout    = typename traits::array_layout;
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
      Morpheus::DenseVector<index_type, index_type, array_layout,
                            execution_space, memory_traits>;
  using index_array_pointer = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;

  /*! The type of \p DenseVector that holds the value_type data */
  using value_array_type =
      Morpheus::DenseVector<value_type, index_type, array_layout,
                            execution_space, memory_traits>;
  using value_array_pointer = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;

  /**
   * @brief The default destructor.
   */
  ~CooMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a CooMatrix container
   * from another CooMatrix container with the same properties.
   */
  CooMatrix(const CooMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a CooMatrix container
   * from another CooMatrix container with the same properties.
   */
  CooMatrix(CooMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a CooMatrix container
   * from another CooMatrix container with the same properties.
   */
  CooMatrix &operator=(const CooMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a CooMatrix container
   * from another CooMatrix container with the same properties.
   */
  CooMatrix &operator=(CooMatrix &&) = default;

  /**
   * @brief Construct an empty CooMatrix object
   */
  inline CooMatrix() : base(), _row_indices(), _column_indices(), _values() {}

  /**
   * @brief Construct a CooMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   */
  inline CooMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries)
      : base(num_rows, num_cols, num_entries),
        _row_indices(num_entries),
        _column_indices(num_entries),
        _values(num_entries) {}

  // Construct from raw pointers
  template <typename ValuePtr, typename IndexPtr>
  explicit inline CooMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexPtr rind_ptr, IndexPtr cind_ptr,
      ValuePtr vals_ptr,
      typename std::enable_if<
          (std::is_pointer<ValuePtr>::value &&
           is_same_value_type<value_type, ValuePtr>::value &&
           memory_traits::is_unmanaged) &&
          (std::is_pointer<IndexPtr>::value &&
           is_same_index_type<index_type, IndexPtr>::value &&
           memory_traits::is_unmanaged)>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_indices(size_t(num_entries), rind_ptr),
        _column_indices(size_t(num_entries), cind_ptr),
        _values(size_t(num_entries), vals_ptr) {}

  /**
   * @brief Construct a CooMatrix object with shape (num_rows, num_cols) and
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
  explicit inline CooMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexArray rind, IndexArray cind,
      ValueArray vals,
      typename std::enable_if<
          is_dense_vector_format_container<ValueArray>::value &&
          is_dense_vector_format_container<IndexArray>::value &&
          is_compatible<typename CooMatrix::value_array_type,
                        ValueArray>::value &&
          is_compatible<typename CooMatrix::index_array_type,
                        IndexArray>::value &&
          !ValueArray::memory_traits::is_unmanaged &&
          !IndexArray::memory_traits::is_unmanaged>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _row_indices(rind),
        _column_indices(cind),
        _values(vals) {}

  /**
   * @brief Constructs a CooMatrix from another compatible CooMatrix
   *
   * @par Constructs a CooMatrix from another compatible CooMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  CooMatrix(const CooMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                CooMatrix, CooMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _row_indices(src.crow_indices()),
        _column_indices(src.ccolumn_indices()),
        _values(src.cvalues()) {}

  /**
   * @brief Assigns a CooMatrix from another compatible CooMatrix
   *
   * @par Overview
   * Assigns a CooMatrix from another compatible CooMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<CooMatrix, CooMatrix<VR, PR...>>::value,
      CooMatrix &>::type
  operator=(const CooMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _row_indices    = src.crow_indices();
    _column_indices = src.ccolumn_indices();
    _values         = src.cvalues();

    return *this;
  }

  /**
   * @brief Constructs a CooMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a CooMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  CooMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_dynamically_compatible<
                CooMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  /**
   * @brief Assigns a CooMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a CooMatrix from a compatible DynamicMatrix i.e a matrix that
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
      is_dynamically_compatible<CooMatrix, DynamicMatrix<VR, PR...>>::value,
      CooMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  /**
   * @brief Construct a CooMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  CooMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assign to CooMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes CooMatrix with shape of (num_rows, num_cols) and sets number
   * of non-zero entries to num_entries.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_entries Number of non-zero entries in resized matrix.
   */
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries) {
    base::resize(num_rows, num_cols, num_entries);
    _row_indices.resize(num_entries);
    _column_indices.resize(num_entries);
    _values.resize(num_entries);
  }

  /**
   * @brief Resizes CooMatrix with the shape and number of non-zero entries of
   * another CooMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source CooMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const CooMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
  }

  /**
   * @brief Allocates memory from another CooMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p CooMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline CooMatrix &allocate(const CooMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz());
    return *this;
  }

  /**
   * @brief Sorts matrix elements by row index
   *
   */
  void sort_by_row(void) {
    throw Morpheus::NotImplementedException("CooMatrix.sort_by_row()");
  }

  /**
   * @brief Sorts matrix elements by row index first and then by column index.
   *
   */
  void sort(void) { Morpheus::sort_by_row_and_column<execution_space>(*this); }

  /**
   * @brief Determines whether matrix elements are sorted by row index
   *
   */
  bool is_sorted_by_row(void) {
    throw Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row()");
    return true;
  }

  /**
   * @brief Determines whether matrix elements are sorted by row and column
   * index
   *
   */
  bool is_sorted(void) { return Morpheus::is_sorted<execution_space>(*this); }

  /**
   * @brief Returns the format enum assigned to the CooMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * CooMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

  /**
   * @brief Returns a reference to the row index of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Row index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  row_indices(index_type n) {
    return _row_indices(n);
  }

  /**
   * @brief Returns a reference to the column index of the matrix with index \p
   * n
   *
   * @param n Index of the value to extract
   * @return Column index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  column_indices(index_type n) {
    return _column_indices(n);
  }

  /**
   * @brief Returns a reference to the value of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(index_type n) {
    return _values(n);
  }

  /**
   * @brief Returns a const-reference to the row index of the matrix with index
   * \p n
   *
   * @param n Index of the value to extract
   * @return Row index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_reference
  crow_indices(index_type n) const {
    return _row_indices(n);
  }

  /**
   * @brief Returns a const-reference to the column index of the matrix with
   * index \p n
   *
   * @param n Index of the value to extract
   * @return Column index at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_reference
  ccolumn_indices(index_type n) const {
    return _column_indices(n);
  }

  /**
   * @brief Returns a const-reference to the value of the matrix with index \p n
   *
   * @param n Index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const value_array_reference
  cvalues(index_type n) const {
    return _values(n);
  }

  /**
   * @brief Returns a reference to the row indices of the matrix.
   *
   * @return index_array_type&  A reference to the row indices.
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &row_indices() {
    return _row_indices;
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
   * @brief Returns a const-reference to the row indices of the matrix.
   *
   * @return const index_array_type&  A const reference to the row indices.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const index_array_type &crow_indices() const {
    return _row_indices;
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
  index_array_type _row_indices, _column_indices;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::COO_FORMAT;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_COOMATRIX_HPP