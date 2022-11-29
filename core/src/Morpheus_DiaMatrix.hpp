/**
 * Morpheus_DiaMatrix.hpp
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

#ifndef MORPHEUS_DIAMATRIX_HPP
#define MORPHEUS_DIAMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_DenseVector.hpp>
#include <Morpheus_DenseMatrix.hpp>
#include <Morpheus_DynamicMatrix.hpp>
#include <Morpheus_MatrixBase.hpp>

#include <impl/Dia/Morpheus_Utils_Impl.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the Diagonal (DIA) Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The DiaMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container stores the contents of a sparse matrix along
 * its diagonals allowing for very quick arithmetic operations for matrices with
 * very few non-zero diagonals. However, in the case if the number of non-zero
 * diagonals becomes very large the performance is hindered because of the
 * excessive zero padding required. It is a polymorphic container in the sense
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
 *  using Matrix = Morpheus::DiaMatrix<double, Kokkos::HostSpace>;
 *  using index_array_type = typename Matrix::index_array_type;
 *  using value_array_type = typename Matrix::value_array_type;
 *
 *  index_array_type offsets(4, 0);
 *  value_array_type values(4,0);
 *
 *  offset[0] = -1; offset[1] = 0; offset[2] = 1; offset[3] = 2;
 * // [* 1 0 2]
 * // [0 0 3 *]
 * // [4 0 * *]
 *  value(0,0) = 0; value(1,0) = 0; value(2,0) = 4;
 *  value(0,1) = 1; value(1,1) = 0; value(2,1) = 0;
 *  value(0,2) = 0; value(1,2) = 3; value(2,2) = 0;
 *  value(0,3) = 2; value(1,3) = 0; value(2,3) = 0;
 *
 *  // Construct the matrix from diagonal offsets and values
 *  Matrix A(3, 3, 4, offset, values);
 *
 *  Morpheus::print(A); // prints A
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class DiaMatrix : public MatrixBase<DiaMatrix, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = ContainerTraits<DiaMatrix, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  using base = MatrixBase<DiaMatrix, ValueType, Properties...>;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<DiaFormatTag>::tag;

  /*! The type of the values held by the container - can be const */
  using value_type = typename traits::value_type;
  /*! The non-constant type of the values held by the container */
  using non_const_value_type = typename traits::non_const_value_type;
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
      Morpheus::DenseVector<index_type, index_type, array_layout, backend,
                            memory_traits>;
  using const_index_array_type = const index_array_type;
  using index_array_pointer    = typename index_array_type::value_array_pointer;
  using index_array_reference =
      typename index_array_type::value_array_reference;
  using const_index_array_reference = const index_array_reference;

  /*! The type of \p DenseMatrix that holds the value_type data */
  using value_array_type =
      Morpheus::DenseMatrix<value_type, index_type, array_layout, backend,
                            memory_traits>;
  using const_value_array_type = const value_array_type;
  using value_array_pointer    = typename value_array_type::value_array_pointer;
  using value_array_reference =
      typename value_array_type::value_array_reference;
  using const_value_array_reference = const value_array_reference;

  /**
   * @brief The default destructor.
   */
  ~DiaMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a DiaMatrix container
   * from another DiaMatrix container with the same properties.
   */
  DiaMatrix(const DiaMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a DiaMatrix container
   * from another DiaMatrix container with the same properties.
   */
  DiaMatrix(DiaMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a DiaMatrix container
   * from another DiaMatrix container with the same properties.
   */
  DiaMatrix &operator=(const DiaMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a DiaMatrix container
   * from another DiaMatrix container with the same properties.
   */
  DiaMatrix &operator=(DiaMatrix &&) = default;

  /**
   * @brief Construct an empty DiaMatrix object
   */
  inline DiaMatrix()
      : base(), _ndiags(0), _alignment(0), _diagonal_offsets(), _values() {}

  /**
   * @brief Construct a DiaMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries concentrated around num_diagonals
   * diagonals.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   * @param num_diagonals Number of occupied diagonals
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline DiaMatrix(const index_type num_rows, const index_type num_cols,
                   const index_type num_entries, const index_type num_diagonals,
                   const index_type alignment = 32)
      : base(num_rows, num_cols, num_entries),
        _ndiags(num_diagonals),
        _alignment(alignment),
        _diagonal_offsets(num_diagonals) {
    _values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                   num_diagonals);
  }

  /**
   * @brief Construct a DiaMatrix object with shape (num_rows, num_cols) and
   * number of non-zeros equal to num_entries and assign the diagonal offsets
   * and values from \p DenseVector and \p DenseMatrix arrays equivalently.
   *
   * @tparam ValueArray Value type DenseMatrix type.
   * @tparam IndexArray Index type DenseVector type.
   *
   * @param num_rows  Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_entries Number of non-zero values in the matrix.
   * @param diag_offsets \p DenseVector containing the diagonal offsets of the
   * matrix.
   * @param vals \p DenseMatrix containing the values of the matrix.
   */
  template <typename ValueArray, typename IndexArray>
  explicit inline DiaMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, const IndexArray &diag_offsets,
      const ValueArray &vals,
      typename std::enable_if<
          is_dense_matrix_format_container<ValueArray>::value &&
          is_dense_vector_format_container<IndexArray>::value &&
          is_compatible<typename DiaMatrix::value_array_type,
                        ValueArray>::value &&
          is_compatible<typename DiaMatrix::index_array_type,
                        IndexArray>::value &&
          !ValueArray::memory_traits::is_unmanaged &&
          !IndexArray::memory_traits::is_unmanaged>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _diagonal_offsets(diag_offsets),
        _values(vals) {
    _ndiags    = _diagonal_offsets.size();
    _alignment = _values.nrows();
  }

  // Construct from raw pointers
  template <typename ValuePtr, typename IndexPtr>
  explicit inline DiaMatrix(
      const index_type num_rows, const index_type num_cols,
      const index_type num_entries, IndexPtr diag_offsets_ptr,
      ValuePtr vals_ptr, const index_type num_diagonals,
      const index_type alignment = 32,
      typename std::enable_if<
          (std::is_pointer<ValuePtr>::value &&
           is_same_value_type<value_type, ValuePtr>::value &&
           memory_traits::is_unmanaged) &&
          (std::is_pointer<IndexPtr>::value &&
           is_same_index_type<index_type, IndexPtr>::value &&
           memory_traits::is_unmanaged)>::type * = nullptr)
      : base(num_rows, num_cols, num_entries),
        _diagonal_offsets(num_diagonals, diag_offsets_ptr),
        _values(Impl::get_pad_size<index_type>(num_rows, alignment),
                num_diagonals, vals_ptr) {}

  /**
   * @brief Constructs a DiaMatrix from another compatible DiaMatrix
   *
   * @par Constructs a DiaMatrix from another compatible DiaMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  DiaMatrix(const DiaMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                DiaMatrix, DiaMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _ndiags(src.ndiags()),
        _alignment(src.alignment()),
        _diagonal_offsets(src.cdiagonal_offsets()),
        _values(src.cvalues()) {}

  /**
   * @brief Assigns a DiaMatrix from another compatible DiaMatrix
   *
   * @par Overview
   * Assigns a DiaMatrix from another compatible DiaMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<DiaMatrix, DiaMatrix<VR, PR...>>::value,
      DiaMatrix &>::type
  operator=(const DiaMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _ndiags           = src.ndiags();
    _alignment        = src.alignment();
    _diagonal_offsets = src.cdiagonal_offsets();
    _values           = src.cvalues();

    return *this;
  }

  /**
   * @brief Constructs a DiaMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a DiaMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  DiaMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_dynamically_compatible<
                DiaMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  /**
   * @brief Assigns a DiarMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a DiarMatrix from a compatible DynamicMatrix i.e a matrix that
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
      is_dynamically_compatible<DiaMatrix, DynamicMatrix<VR, PR...>>::value,
      DiaMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  /**
   * @brief Construct a DiaMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  DiaMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assigns to DiaMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes DiaMatrix with shape of (num_rows, num_cols) and sets number
   * of non-zero entries to num_entries, number of diagonals to num_diagonals
   * and the alignment.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_entries Number of non-zero entries in resized matrix.
   * @param num_diagonals Number of occupied diagonals in resized matrix.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline void resize(const index_type num_rows, const index_type num_cols,
                     const index_type num_entries,
                     const index_type num_diagonals,
                     const index_type alignment = 32) {
    base::resize(num_rows, num_cols, num_entries);
    _ndiags    = num_diagonals;
    _alignment = alignment;

    if (Impl::exceeds_tolerance(num_rows, num_entries, _ndiags)) {
      throw Morpheus::FormatConversionException(
          "DiaMatrix fill-in would exceed maximum tolerance");
    }

    _diagonal_offsets.resize(num_diagonals);
    _values.resize(Impl::get_pad_size<index_type>(num_rows, alignment),
                   num_diagonals);
  }

  /**
   * @brief Resizes DiaMatrix with the shape and number of non-zero entries of
   * another DiaMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source CsrMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const DiaMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz(), src.ndiags(), src.alignment());
  }

  /**
   * @brief Allocates memory from another DiaMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p DiaMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline DiaMatrix &allocate(const DiaMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.nnnz(), src.ndiags(), src.alignment());
    return *this;
  }

  /**
   * @brief Returns the format enum assigned to the DiaMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * DiaMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

  /**
   * @brief Returns a reference to the diagonal offsets of the matrix with index
   * \p n
   *
   * @param n Index of the value to extract
   * @return Diagonal offset at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_reference
  diagonal_offsets(index_type n) {
    return _diagonal_offsets(n);
  }

  /**
   * @brief Returns a reference to the value of the matrix with indexes (i, j)
   *
   * @param i Row index of the value to extract
   * @param j Column index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference values(index_type i,
                                                             index_type j) {
    return _values(i, j);
  }

  /**
   * @brief Returns a const-reference to the diagonal offsets of the matrix with
   * index \p n
   *
   * @param n Index of the value to extract
   * @return Diagonal offset at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_reference
  cdiagonal_offsets(index_type n) const {
    return _diagonal_offsets(n);
  }

  /**
   * @brief Returns a const-reference to the value of the matrix with indexes
   * (i, j)
   *
   * @param i Row index of the value to extract
   * @param j Column index of the value to extract
   * @return Value of the element at index \p n
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_reference
  cvalues(index_type i, index_type j) const {
    return _values(i, j);
  }

  /**
   * @brief Returns a reference to the diagonal offsets of the matrix.
   *
   * @return index_array_type&  A reference to the diagonal offsets.
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_array_type &diagonal_offsets() {
    return _diagonal_offsets;
  }

  /**
   * @brief Returns a reference to the values of the matrix.
   *
   * @return value_array_type&  A reference to the values.
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_type &values() { return _values; }

  /**
   * @brief Returns a const-reference to the diagonal offsets of the matrix.
   *
   * @return index_array_type&  A reference to the diagonal offsets.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_index_array_type &cdiagonal_offsets()
      const {
    return _diagonal_offsets;
  }

  /**
   * @brief Returns a const-reference to the values of the matrix.
   *
   * @return value_array_type&  A reference to the values.
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_value_array_type &cvalues() const {
    return _values;
  }

  /**
   * @brief Returns the number of occupied diagonals of the matrix.
   *
   * @return index_type Number of occupied diagonals
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_type ndiags() const { return _ndiags; }

  /**
   * @brief Returns the amount of padding used to align the data.
   *
   * @return index_type Amount of padding used to align the data
   */
  MORPHEUS_FORCEINLINE_FUNCTION index_type alignment() const {
    return _alignment;
  }

  /**
   * @brief Sets the number of occupied diagonals of the matrix.
   *
   * @param num_diagonals number of new diagonals
   */
  MORPHEUS_FORCEINLINE_FUNCTION void set_ndiags(
      const index_type num_diagonals) {
    _ndiags = num_diagonals;
  }

  /**
   * @brief Sets amount of padding with which to align the data.
   *
   * @param alignment New amount of padding.
   */
  MORPHEUS_FORCEINLINE_FUNCTION void set_alignment(const index_type alignment) {
    _alignment = alignment;
  }

 private:
  index_type _ndiags, _alignment;
  index_array_type _diagonal_offsets;
  value_array_type _values;
  static constexpr formats_e _id = Morpheus::DIA_FORMAT;
};

/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_DIAMATRIX_HPP