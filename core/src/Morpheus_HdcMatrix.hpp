/**
 * Morpheus_HdcMatrix.hpp
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

#ifndef MORPHEUS_HDCMATRIX_HPP
#define MORPHEUS_HDCMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_MatrixBase.hpp>
#include <Morpheus_CsrMatrix.hpp>
#include <Morpheus_DiaMatrix.hpp>
// #include <Morpheus_DynamicMatrix.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \brief Two-dimensional Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the Hybrid Diagonal Compressed Sparse Row (HDC)
 * Sparse Matrix Format Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The HdcMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container is the implementation of the HDC Format.
 * The HDC representation combines the DIA and CSR formats to balance out
 * the inefficiencies of storing irregular matrices in DIA format. In other
 * words, HDC separates the dense diagonal region and stores it in DIA format,
 * from the other parts that are stored in CSR. It is a polymorphic container in
 * the sense that it can store scalar or integer type values, on host or device
 * depending how the template parameters are selected.
 *
 * \note HDC matrices are not intended to be used to construct the matrix
 * directly. Instead, they are used to convert from/to another format.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 * // Matrix to Build
 * // [1 * * 2]
 * // [* 3 4 *]
 * // [* 5 6 *]
 * // [7 * * 8]
 * int main(){
 *  using Matrix = Morpheus::HdcMatrix<double, Kokkos::HostSpace>;
 *  using csr_matrix_type = typename Matrix::csr_matrix_type;
 *  using dia_matrix_type = typename Matrix::dia_matrix_type;
 *  using size_type = typename Matrix::size_type;
 *
 *  csr_matrix_type csr(4, 4, 4);
 *  dia_matrix_type dia(4, 4, 4, 1, 32);
 *
 *  dia.diagonal_offsets(0) = 0; // Main Diagonal
 *  dia.values(0,0) = 1;
 *  dia.values(1,0) = 3;
 *  dia.values(2,0) = 6;
 *  dia.values(3,0) = 8;
 *
 *  csr.row_offsets(0) = 0;
 *  csr.row_offsets(1) = 1;
 *  csr.row_offsets(2) = 2;
 *  csr.row_offsets(3) = 3;
 *  csr.row_offsets(4) = 4;
 *  csr.column_indices(0) = 3; csr.values(0) = 2;
 *  csr.column_indices(1) = 2; csr.values(1) = 4;
 *  csr.column_indices(2) = 1; csr.values(2) = 5;
 *  csr.column_indices(3) = 0; csr.values(3) = 7;
 *
 *  // Construct the matrix from DIA and CSR parts
 *  Matrix A(dia, csr);
 *
 *  Morpheus::print(A); // prints A
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class HdcMatrix : public MatrixBase<HdcMatrix, ValueType, Properties...> {
 public:
  //!< The traits associated with the particular container
  using traits = ContainerTraits<HdcMatrix, ValueType, Properties...>;
  //!< The complete type of the container
  using type = typename traits::type;
  using base = MatrixBase<HdcMatrix, ValueType, Properties...>;
  //!< The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::HdcFormatTag>::tag;

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

  using csr_matrix_type = Morpheus::CsrMatrix<ValueType, Properties...>;
  using dia_matrix_type = Morpheus::DiaMatrix<ValueType, Properties...>;
  using const_csr_matrix_type =
      const Morpheus::CsrMatrix<ValueType, Properties...>;
  using const_dia_matrix_type =
      const Morpheus::DiaMatrix<ValueType, Properties...>;

  /**
   * @brief The default destructor.
   */
  ~HdcMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a HdcMatrix container
   * from another HdcMatrix container with the same properties.
   */
  HdcMatrix(const HdcMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a HdcMatrix container
   * from another HdcMatrix container with the same properties.
   */
  HdcMatrix(HdcMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a HdcMatrix container
   * from another HdcMatrix container with the same properties.
   */
  HdcMatrix &operator=(const HdcMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a HdcMatrix container
   * from another HdcMatrix container with the same properties.
   */
  HdcMatrix &operator=(HdcMatrix &&) = default;

  /**
   * @brief Construct an empty HdcMatrix object
   */
  inline HdcMatrix() : base(), _dia(), _csr(), _alignment() {}

  /**
   * @brief Construct a HdcMatrix object from shape.
   *
   * @param num_rows Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_dia_entries Number of non-zeros in the DIA part.
   * @param num_csr_entries Number of non-zeros in the CSR part.
   * @param num_diagonals Number of diagonals in the DIA part.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline HdcMatrix(const size_type num_rows, const size_type num_cols,
                   const size_type num_dia_entries,
                   const size_type num_csr_entries,
                   const size_type num_diagonals,
                   const size_type alignment = 32)
      : base(num_rows, num_cols, num_dia_entries + num_csr_entries),
        _dia(num_rows, num_cols, num_dia_entries, num_diagonals, alignment),
        _csr(num_rows, num_cols, num_csr_entries),
        _alignment(alignment) {}

  // TODO: Construct from pointers

  /**
   * @brief Construct a HdcMatrix object from the CSR and DIA parts.
   *
   * @tparam DiaMatrixType Type of the \p DiaMatrix.
   * @tparam CsrMatrixType Type of the \p CsrMatrix.
   *
   * @param dia \p DiaMatrix with the values of the DIA part of the matrix.
   * @param csr \p CsrMatrix with the values of the CSR part of the matrix.
   */
  template <typename DiaMatrixType, typename CsrMatrixType>
  explicit inline HdcMatrix(
      const DiaMatrixType &dia, const CsrMatrixType &csr,
      typename std::enable_if<
          is_dia_matrix_format_container<DiaMatrixType>::value &&
          is_csr_matrix_format_container<CsrMatrixType>::value &&
          is_compatible<typename HdcMatrix::dia_matrix_type,
                        DiaMatrixType>::value &&
          is_compatible<typename HdcMatrix::csr_matrix_type,
                        CsrMatrixType>::value &&
          !HdcMatrix::dia_matrix_type::traits::memory_traits::is_unmanaged &&
          !HdcMatrix::csr_matrix_type::traits::memory_traits::is_unmanaged>::
          type * = nullptr)
      : base(dia.nrows(), dia.ncols(), dia.nnnz() + csr.nnnz()),
        _dia(dia),
        _csr(csr),
        _alignment(cdia().alignment()) {}

  /**
   * @brief Constructs a HdcMatrix from another compatible HdcMatrix
   *
   * @par Constructs a HdcMatrix from another compatible HdcMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  HdcMatrix(const HdcMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                HdcMatrix, HdcMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _dia(src.cdia()),
        _csr(src.ccsr()),
        _alignment(src.alignment()) {}

  /**
   * @brief Assigns a HdcMatrix from another compatible HdcMatrix
   *
   * @par Overview
   * Assigns a HdcMatrix from another compatible HdcMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<HdcMatrix, HdcMatrix<VR, PR...>>::value,
      HdcMatrix &>::type
  operator=(const HdcMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _dia       = src.cdia();
    _csr       = src.ccsr();
    _alignment = src.alignment();

    return *this;
  }

  /**
   * @brief Constructs a HdcMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a HdcMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  HdcMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_dynamically_compatible<
                HdcMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  /**
   * @brief Assigns a HdcMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a HdcMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this  will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_dynamically_compatible<HdcMatrix, DynamicMatrix<VR, PR...>>::value,
      HdcMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  /**
   * @brief Construct a HdcMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  HdcMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assign to HdcMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes HdcMatrix from shape.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_dia_entries Number of non-zeros in the DIA part.
   * @param num_csr_entries Number of non-zeros in the CSR part.
   * @param num_diagonals Number of diagonals in the DIA part.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline void resize(const size_type num_rows, const size_type num_cols,
                     const size_type num_dia_entries,
                     const size_type num_csr_entries,
                     const size_type num_diagonals,
                     const size_type alignment = 32) {
    base::resize(num_rows, num_cols, num_dia_entries + num_csr_entries);

    _dia.resize(num_rows, num_cols, num_dia_entries, num_diagonals, alignment);
    _csr.resize(num_rows, num_cols, num_csr_entries);

    _alignment = alignment;
  }

  /**
   * @brief Resizes HdcMatrix with the shape and number of non-zero entries of
   * another HdcMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source HdcMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const HdcMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.cdia().nnnz(), src.ccsr().nnnz(),
           src.cdia().ndiags(), src.alignment());
  }

  /**
   * @brief Allocates memory from another HdcMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p HdcMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline HdcMatrix &allocate(const HdcMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.cdia().nnnz(), src.ccsr().nnnz(),
           src.cdia().ndiags(), src.alignment());
    return *this;
  }

  /**
   * @brief Returns the format enum assigned to the HdcMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * HdcMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

  /**
   * @brief Returns a reference to the DIA part of the matrix
   *
   * @return dia_matrix_type& A reference to the DiaMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION dia_matrix_type &dia() { return _dia; }

  /**
   * @brief Returns a reference to the CSR part of the matrix
   *
   * @return csr_matrix_type& A reference to the CsrMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION csr_matrix_type &csr() { return _csr; }

  /**
   * @brief Returns a const-reference to the DIA part of the matrix
   *
   * @return const dia_matrix_type& A const-reference to the DiaMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION const dia_matrix_type &cdia() const {
    return _dia;
  }

  /**
   * @brief Returns a const-reference to the CSR part of the matrix
   *
   * @return const csr_matrix_type& A const-reference to the CsrMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION const csr_matrix_type &ccsr() const {
    return _csr;
  }

  /**
   * @brief Returns the amount of padding used to align the data.
   *
   * @return size_type Amount of padding used to align the data
   */
  MORPHEUS_FORCEINLINE_FUNCTION size_type alignment() const {
    return _alignment;
  }

  /**
   * @brief Sets amount of padding with which to align the data.
   *
   * @param alignment New amount of padding.
   */
  MORPHEUS_FORCEINLINE_FUNCTION void set_alignment(const size_type alignment) {
    _alignment = alignment;
  }

 private:
  dia_matrix_type _dia;
  csr_matrix_type _csr;
  size_type _alignment;
  static constexpr formats_e _id = Morpheus::HDC_FORMAT;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_HDCMATRIX_HPP