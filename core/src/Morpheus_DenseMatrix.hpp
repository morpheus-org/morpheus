/**
 * Morpheus_DenseMatrix.hpp
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

#ifndef MORPHEUS_DENSEMATRIX_HPP
#define MORPHEUS_DENSEMATRIX_HPP

#include <Morpheus_Exceptions.hpp>
#include <Morpheus_FormatTags.hpp>
#include <Morpheus_MatrixBase.hpp>

#include <impl/Morpheus_Functors.hpp>

#include <Kokkos_Core.hpp>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief The DenseMatrix container is a two-dimensional dense container that
 * contains contiguous elements. It is a polymorphic container in the sense that
 * it can store scalar or integer type values, on host or device depending how
 * the template parameters are selected.
 *
 * @tparam ValueType type of values to store
 * @tparam Properties optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * TODO
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 *
 * int main(){
 *  // Construct a vector on host, of size 10 and with values set to 5.0
 *  Morpheus::DenseMatrix<double, Kokkos::HostSpace> A(10, 10, 5.0);
 *
 *  // Set some values
 *  A(2,4) = 5.0;
 *  A(5,4) = -2.0;
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class DenseMatrix : public MatrixBase<DenseMatrix, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = ContainerTraits<DenseMatrix, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  using base = MatrixBase<DenseMatrix, ValueType, Properties...>;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::DenseMatrixFormatTag>::tag;

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

  /*! The type of view that holds the value_type data */
  using value_array_type =
      Kokkos::View<value_type **, array_layout, execution_space, memory_traits>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

  /**
   * @brief Default destructor.
   */
  ~DenseMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a DenseMatrix
   * container from another DenseMatrix container with the same properties.
   */
  DenseMatrix(const DenseMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a DenseMatrix
   * container from another DenseMatrix container with the same properties.
   */
  DenseMatrix(DenseMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a DenseMatrix
   * container from another DenseMatrix container with the same properties.
   */
  DenseMatrix &operator=(const DenseMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a DenseMatrix
   * container from another DenseMatrix container with the same properties.
   */
  DenseMatrix &operator=(DenseMatrix &&) = default;

  /**
   * @brief Construct an empty DenseVector object
   */
  inline DenseMatrix() : base(), _values() {}

  /**
   * @brief Construct a DenseMatrix object with shape (num_rows, num_cols) and
   * values set to \p val
   *
   * @param num_rows Number of rows
   * @param num_cols Number of columns
   * @param val Value at which the elements of the DenseMatrix will be set to
   */
  inline DenseMatrix(const index_type num_rows, const index_type num_cols,
                     const value_type val = 0)
      : base(num_rows, num_cols, num_rows * num_cols),
        _values("matrix", size_t(num_rows), size_t(num_cols)) {
    assign(num_rows, num_cols, val);
  }

  /**
   * @brief Construct a DenseMatrix object from a raw pointer. This is only
   * enabled if the DenseMatrix is an unmanaged container.
   *
   * @tparam ValuePtr Pointer type
   * @param num_rows Number of rows
   * @param num_cols Number of columns
   * @param ptr Pointer value
   */
  template <typename ValuePtr>
  explicit DenseMatrix(
      const index_type num_rows, const index_type num_cols, ValuePtr ptr,
      typename std::enable_if<std::is_pointer<ValuePtr>::value &&
                              is_same_value_type<value_type, ValuePtr>::value &&
                              memory_traits::is_unmanaged>::type * = nullptr)
      : base(num_rows, num_cols, num_rows * num_cols),
        _values(ptr, size_t(num_rows), size_t(num_cols)) {
    static_assert(std::is_same<value_array_pointer, ValuePtr>::value,
                  "Constructing DenseMatrix to wrap user memory must supply "
                  "matching pointer type");
  }

  /**
   * @brief Shallow Copy contrustor from another DenseMatrix container with
   * different properties. Note that this is only possible when the \p
   * is_compatible check is satisfied.
   *
   * @tparam VR Value Type of the container we are constructing from.
   * @tparam PR Optional properties of the container we are constructing from.
   * @param src The \p DenseMatrix container we are constructing from.
   */
  template <class VR, class... PR>
  inline DenseMatrix(
      const DenseMatrix<VR, PR...> &src,
      typename std::enable_if<is_format_compatible<
          DenseMatrix, DenseMatrix<VR, PR...>>::value>::type * = nullptr)
      : base(src.nrows(), src.ncols(), src.nrows() * src.ncols()),
        _values(src.const_view()) {}

  /**
   * @brief Shallow Copy Assignment from another DenseMatrix container with
   * different properties. Note that this is only possible when the \p
   * is_compatible check is satisfied.
   *
   * @tparam VR Value Type of the container we are copying from.
   * @tparam PR Optional properties of the container we are copying from.
   * @param src The \p DenseMatrix container we are copying from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<DenseMatrix, DenseMatrix<VR, PR...>>::value,
      DenseMatrix &>::type
  operator=(const DenseMatrix<VR, PR...> &src) {
    this->set_nrows(src.nrows());
    this->set_ncols(src.ncols());
    this->set_nnnz(src.nnnz());
    _values = src.const_view();
    return *this;
  }

  /**
   * @brief Construct a DenserMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <class MatrixType>
  DenseMatrix(const MatrixType &src) = delete;

  /**
   * @brief Assign to DenseMatrix object from another storage format. This
   * functionality is disabled to avoid implicit copies and conversion
   * operations.
   *
   * @tparam MatrixType Any of the supported storage formats.
   * @param src The source container.
   */
  template <class MatrixType>
  reference operator=(const MatrixType &src) = delete;

  /**
   * @brief Assigns (num_rows * num_cols) elements of value \p val to the
   * DenseMatrix. The container will always resize to match the new shape.
   *
   * @param num_rows Number of rows
   * @param num_cols Number of columns
   * @param val Value to assign
   */
  inline void assign(index_type num_rows, index_type num_cols,
                     const value_type val) {
    using range_policy = Kokkos::RangePolicy<index_type, execution_space>;

    range_policy policy(0, num_rows);
    Impl::set_functor<value_array_type, value_type, index_type> f(_values, val,
                                                                  num_cols);
    Kokkos::parallel_for("Morpheus::DenseMatrix::assign", policy, f);
  }

  /**
   * @brief Resizes DenseMatrix with shape (num_rows * num_cols). Overlapping
   * subextents will preserve their contents.
   *
   * @param num_rows Number of new rows
   * @param num_cols Number of new columns
   */
  inline void resize(index_type num_rows, index_type num_cols) {
    base::resize(num_rows, num_cols, num_rows * num_cols);
    Kokkos::resize(_values, size_t(num_rows), size_t(num_cols));
  }

  /**
   * @brief Resizes DenseVector with the shape another DenseMatrix with
   * different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source DenseMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const DenseMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols());
  }

  /**
   * @brief Allocates memory from another DenseMatrix container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p DenseMatrix container we are allocating from.
   */
  template <class VR, class... PR>
  inline DenseMatrix &allocate(const DenseMatrix<VR, PR...> &src) {
    resize(src.nrows(), src.ncols());
    return *this;
  }

  /**
   * @brief Returns a reference to the element with index (i,j)
   *
   * @param i First index of the value to extract
   * @param j Second index of the value to extract
   * @return Element at index (i,j)
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference
  operator()(index_type i, index_type j) const {
    return _values(i, j);
  }

  /**
   * @brief Returns a pointer to the data at the beginning of the container
   *
   * @return Pointer type of the value_type data
   */
  inline value_array_pointer data() const { return _values.data(); }

  /**
   * @brief Returns a reference to the beginning of the view that holds the data
   *
   * @return Type of view that holds the data
   */
  inline value_array_type &view() { return _values; }

  /**
   * @brief Returns a constant reference to the beginning of the view that holds
   * the data
   *
   * @return Constant type of view that holds the data
   */
  inline const value_array_type &const_view() const { return _values; }

 private:
  value_array_type _values;
};

/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEMATRIX_HPP
