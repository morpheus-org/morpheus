/**
 * Morpheus_HybMatrix.hpp
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

#ifndef MORPHEUS_HYBMATRIX_HPP
#define MORPHEUS_HYBMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_MatrixBase.hpp>
#include <Morpheus_CooMatrix.hpp>
#include <Morpheus_EllMatrix.hpp>
#include <Morpheus_DynamicMatrix.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \brief Two-dimensional Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the HYBRID (HYB) Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The HybMatrix container is a two-dimensional container that represents
 * a sparse matrix. This container is the implementation of the HYBRID Format.
 * The HYBRID representation combines the COO and ELL formats to balance out
 * rows with assymetric number of non-zeros. In other words, rows that have up
 * to a max_entries_per_row are stored using ELL format and the rest using COO.
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
 *  using Matrix = Morpheus::HybMatrix<double, Kokkos::HostSpace>;
 *  // TODO
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class HybMatrix : public MatrixBase<HybMatrix, ValueType, Properties...> {
 public:
  //!< The traits associated with the particular container
  using traits = ContainerTraits<HybMatrix, ValueType, Properties...>;
  //!< The complete type of the container
  using type = typename traits::type;
  using base = MatrixBase<HybMatrix, ValueType, Properties...>;
  //!< The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::HybFormatTag>::tag;

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

  using coo_matrix_type = Morpheus::CooMatrix<ValueType, Properties...>;
  using ell_matrix_type = Morpheus::EllMatrix<ValueType, Properties...>;
  using const_coo_matrix_type =
      const Morpheus::CooMatrix<ValueType, Properties...>;
  using const_ell_matrix_type =
      const Morpheus::EllMatrix<ValueType, Properties...>;

  /**
   * @brief The default destructor.
   */
  ~HybMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a HybMatrix container
   * from another HybMatrix container with the same properties.
   */
  HybMatrix(const HybMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a HybMatrix container
   * from another HybMatrix container with the same properties.
   */
  HybMatrix(HybMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a HybMatrix container
   * from another HybMatrix container with the same properties.
   */
  HybMatrix &operator=(const HybMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a HybMatrix container
   * from another HybMatrix container with the same properties.
   */
  HybMatrix &operator=(HybMatrix &&) = default;

  /**
   * @brief Construct an empty HybMatrix object
   */
  inline HybMatrix() : base(), _ell(), _coo(), _alignment() {}

  /**
   * @brief Construct a HybMatrix object from shape.
   *
   * @param num_rows Number of rows of the matrix.
   * @param num_cols Number of columns of the matrix.
   * @param num_ell_entries Number of non-zeros in the ELL part.
   * @param num_coo_entries Number of non-zeros in the COO part.
   * @param num_entries_per_row Max number of non-zeros per row in the ELL part.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline HybMatrix(const size_type num_rows, const size_type num_cols,
                   const size_type num_ell_entries,
                   const size_type num_coo_entries,
                   const size_type num_entries_per_row = 0,
                   const size_type alignment           = 32)
      : base(num_rows, num_cols, num_ell_entries + num_coo_entries),
        _ell(),
        _coo(num_rows, num_cols, num_coo_entries),
        _alignment(alignment) {
    size_type entries_per_row = num_entries_per_row;
    if (num_entries_per_row == 0) {
      entries_per_row = (num_ell_entries + num_coo_entries) / num_rows;
    }

    _ell.resize(num_rows, num_cols, num_ell_entries, entries_per_row,
                alignment);
  }

  // TODO: Construct from pointers

  /**
   * @brief Construct a HybMatrix object from the COO and ELL parts.
   *
   * @tparam EllMatrixType Type of the \p EllMatrix.
   * @tparam CooMatrixType Type of the \p CooMatrix.
   *
   * @param ell \p EllMatrix with the values of the ELL part of the matrix.
   * @param coo \p CooMatrix with the values of the COO part of the matrix.
   */
  template <typename EllMatrixType, typename CooMatrixType>
  explicit inline HybMatrix(
      const EllMatrixType &ell, const CooMatrixType &coo,
      typename std::enable_if<
          is_ell_matrix_format_container<EllMatrixType>::value &&
          is_coo_matrix_format_container<CooMatrixType>::value &&
          is_compatible<typename HybMatrix::ell_matrix_type,
                        EllMatrixType>::value &&
          is_compatible<typename HybMatrix::coo_matrix_type,
                        CooMatrixType>::value &&
          !HybMatrix::ell_matrix_type::traits::memory_traits::is_unmanaged &&
          !HybMatrix::coo_matrix_type::traits::memory_traits::is_unmanaged>::
          type * = nullptr)
      : base(ell.nrows(), ell.ncols(), ell.nnnz() + coo.nnnz()),
        _ell(ell),
        _coo(coo),
        _alignment(cell().alignment()) {}

  /**
   * @brief Constructs a HybMatrix from another compatible HybMatrix
   *
   * @par Constructs a HybMatrix from another compatible HybMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  HybMatrix(const HybMatrix<VR, PR...> &src,
            typename std::enable_if<is_format_compatible<
                HybMatrix, HybMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()),
        _ell(src.cell()),
        _coo(src.ccoo()),
        _alignment(src.alignment()) {}

  /**
   * @brief Assigns a HybMatrix from another compatible HybMatrix
   *
   * @par Overview
   * Assigns a HybMatrix from another compatible HybMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<HybMatrix, HybMatrix<VR, PR...>>::value,
      HybMatrix &>::type
  operator=(const HybMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());

    _ell       = src.cell();
    _coo       = src.ccoo();
    _alignment = src.alignment();

    return *this;
  }

  /**
   * @brief Constructs a HybMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a HybMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the active
   * type of the dynamic matrix is different from the concrete type, this will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  HybMatrix(const DynamicMatrix<VR, PR...> &src,
            typename std::enable_if<is_dynamically_compatible<
                HybMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nnnz()) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());
  }

  /**
   * @brief Assigns a HybMatrix from a compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a HybMatrix from a compatible DynamicMatrix i.e a matrix that
   * satisfies the \p is_dynamically_compatible check. Note that when the
   active
   * type of the dynamic matrix is different from the concrete type, this
   will
   * result in an exception thrown.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_dynamically_compatible<HybMatrix, DynamicMatrix<VR, PR...>>::value,
      HybMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    auto f = std::bind(Impl::any_type_assign(), std::placeholders::_1,
                       std::ref(*this));

    std::visit(f, src.const_formats());

    return *this;
  }

  /**
   * @brief Construct a HybMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  HybMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assign to HybMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <typename MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Resizes HybMatrix from shape.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_ell_entries Number of non-zeros in the ELL part.
   * @param num_coo_entries Number of non-zeros in the COO part.
   * @param num_entries_per_row Max number of non-zeros per row in the ELL part.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline void resize(const size_type num_rows, const size_type num_cols,
                     const size_type num_ell_entries,
                     const size_type num_coo_entries,
                     const size_type num_entries_per_row,
                     const size_type alignment = 32) {
    base::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);

    _ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row,
                alignment);
    _coo.resize(num_rows, num_cols, num_coo_entries);

    _alignment = alignment;
  }

  /**
   * @brief Resizes HybMatrix from shape.
   *
   * @param num_rows Number of rows of resized matrix.
   * @param num_cols Number of columns of resized matrix.
   * @param num_ell_entries Number of non-zeros in the ELL part.
   * @param num_coo_entries Number of non-zeros in the COO part.
   * @param alignment Amount of padding used to align the data (default=32)
   */
  inline void resize(const size_type num_rows, const size_type num_cols,
                     const size_type num_ell_entries,
                     const size_type num_coo_entries,
                     const size_type alignment = 32) {
    size_type avg_entries_per_row =
        (num_ell_entries + num_coo_entries) / num_rows;

    resize(num_rows, num_cols, num_ell_entries, num_coo_entries,
           avg_entries_per_row, alignment);
  }

  /**
   * @brief Resizes HybMatrix with the shape and number of non-zero entries of
   * another HybMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source HybMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const HybMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.cell().nnnz(), src.ccoo().nnnz(),
           src.cell().entries_per_row(), src.alignment());
  }

  /**
   * @brief Allocates memory from another HybMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p HybMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline HybMatrix &allocate(const HybMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols(), src.cell().nnnz(), src.ccoo().nnnz(),
           src.cell().entries_per_row(), src.alignment());
    return *this;
  }

  /**
   * @brief Returns the format enum assigned to the HybMatrix container.
   *
   * @return formats_e The format enum
   */
  formats_e format_enum() const { return _id; }

  /**
   * @brief Returns the equivalent index to the format enum assigned to the
   * HybMatrix container.
   *
   * @return int The equivalent index to \p format_e
   */
  int format_index() const { return static_cast<int>(_id); }

  /**
   * @brief Returns a reference to the ELL part of the matrix
   *
   * @return ell_matrix_type& A reference to the EllMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION ell_matrix_type &ell() { return _ell; }

  /**
   * @brief Returns a reference to the Coo part of the matrix
   *
   * @return coo_matrix_type& A reference to the CooMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION coo_matrix_type &coo() { return _coo; }

  /**
   * @brief Returns a const-reference to the ELL part of the matrix
   *
   * @return const_ell_matrix_type& A const-reference to the EllMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_ell_matrix_type &cell() const {
    return _ell;
  }

  /**
   * @brief Returns a const-reference to the COO part of the matrix
   *
   * @return const_coo_matrix_type& A const-reference to the CooMatrix
   */
  MORPHEUS_FORCEINLINE_FUNCTION const_coo_matrix_type &ccoo() const {
    return _coo;
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
  ell_matrix_type _ell;
  coo_matrix_type _coo;
  size_type _alignment;
  static constexpr formats_e _id = Morpheus::HYB_FORMAT;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_HYBMATRIX_HPP