/**
 * Morpheus_DynamicMatrix.hpp
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

#ifndef MORPHEUS_DYNAMICMATRIX_HPP
#define MORPHEUS_DYNAMICMATRIX_HPP

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_MatrixBase.hpp>

#include <impl/Morpheus_Variant.hpp>
#include <impl/Dynamic/Morpheus_DynamicMatrix_Impl.hpp>

#include <iostream>
#include <string>
#include <functional>

namespace Morpheus {

/**
 * \addtogroup containers_2d 2D Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief Implementation of the Dynamic Sparse Matrix Format
 * Representation.
 *
 * @tparam ValueType Type of values to store
 * @tparam Properties Optional properties to modify the behaviour of the
 * container. Sensible defaults are selected based on the configuration. Please
 * refer to \ref impl/Morpheus_ContainerTraits.hpp to find out more about the
 * valid properties.
 *
 * \par Overview
 * The DynamicMatrix container is a two-dimensional container that acts as a
 * superset of all the available matrix storage formats supported in Morpheus.
 * The purpose of such a container is to enable run-time switching across the
 * different formats under a single unified interface such that we can take
 * advantage of the format that is best suited for the given computation and
 * target hardware. It is a polymorphic container in the sense that it can store
 * scalar or integer type values, on host or device depending how the template
 * parameters are selected.
 *
 * \par Example
 * \code
 * #include <Morpheus_Core.hpp>
 * // Matrix to Build
 * // [1 * 2]
 * // [* * 3]
 * // [* 4 *]
 * int main(){
 *  using DynamicMatrix = Morpheus::DynamicMatrix<double, Kokkos::HostSpace>;
 *  using Matrix = Morpheus::CsrMatrix<double, Kokkos::HostSpace>;
 *  using index_array_type = typename Matrix::index_array_type;
 *  using value_array_type = typename Matrix::value_array_type;
 *
 *  DynamicMatrix A;  // A acts as COO format by default
 *
 *  A.activate(Morpheus::DIA_FORMAT); // Now acts as a DIA format
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
 *  Matrix Acsr(3, 3, 4, off, j, v);
 *
 *  A = Acsr; // Now acts as CSR and holds the same data
 *
 *  Morpheus::print(A); // prints A
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class DynamicMatrix
    : public MatrixBase<DynamicMatrix, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = ContainerTraits<DynamicMatrix, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  using base = MatrixBase<DynamicMatrix, ValueType, Properties...>;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename MatrixFormatTag<Morpheus::DynamicMatrixFormatTag>::tag;

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

  /*! The variant container that holds the various supported containers */
  using variant_type =
      typename MatrixFormats<ValueType, Properties...>::variant;

  /**
   * @brief The default destructor.
   */
  ~DynamicMatrix() = default;
  /**
   * @brief The default copy contructor (shallow copy) of a DynamicMatrix
   * container from another DynamicMatrix container with the same properties.
   */
  DynamicMatrix(const DynamicMatrix &) = default;
  /**
   * @brief The default move contructor (shallow copy) of a DynamicMatrix
   * container from another DynamicMatrix container with the same properties.
   */
  DynamicMatrix(DynamicMatrix &&) = default;
  /**
   * @brief The default copy assignment (shallow copy) of a DynamicMatrix
   * container from another DynamicMatrix container with the same properties.
   */
  DynamicMatrix &operator=(const DynamicMatrix &) = default;
  /**
   * @brief The default move assignment (shallow copy) of a DynamicMatrix
   * container from another DynamicMatrix container with the same properties.
   */
  DynamicMatrix &operator=(DynamicMatrix &&) = default;

  /**
   * @brief Constructs an empty DynamicMatrix object
   */
  inline DynamicMatrix() : _formats() {}

  /**
   * @brief Constructs a DynamicMatrix from another concrete format.
   *
   * @par Overview
   * Constructs a DynamicMatrix from another concrete format supported in
   * Morpheus. For the construction to be valid the \p is_variant_member check
   * needs to be satisfied. In other words, we can only construct from one of
   * the member types in the variant held by the \p DynamicMatrix. Note that the
   * \p DynamicMatrix will change it's active type to the format type of the
   * source Matrix.
   *
   * @tparam Matrix The type of the concrete matrix format.
   * @param src The source container we are constructing from.
   */
  template <typename Matrix>
  inline DynamicMatrix(
      const Matrix &src,
      typename std::enable_if<is_variant_member_v<Matrix, variant_type>>::type
          * = nullptr) {
    this->activate(src.format_enum());
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    auto f = std::bind(Impl::any_type_assign(), std::cref(src),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
  }

  /**
   * @brief Assigns a DynamicMatrix from another concrete format.
   *
   * @par Overview
   * Assigns a DynamicMatrix from another concrete format supported in
   * Morpheus. For the assignment to be valid the \p is_variant_member check
   * needs to be satisfied. In other words, we can only assign from one of
   * the member types in the variant held by the \p DynamicMatrix. Note that the
   * \p DynamicMatrix will change it's active type to the format type of the
   * source Matrix.
   *
   * @tparam Matrix The type of the concrete matrix format.
   * @param src The source container we are assigning from.
   */
  template <typename Matrix>
  typename std::enable_if<is_variant_member_v<Matrix, variant_type>,
                          DynamicMatrix &>::type
  operator=(const Matrix &matrix) {
    this->activate(matrix.format_enum());
    base::resize(matrix.nrows(), matrix.ncols(), matrix.nnnz());

    auto f = std::bind(Impl::any_type_assign(), std::cref(matrix),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
    return *this;
  }

  /**
   * @brief Constructs a DynamicMatrix from another compatible DynamicMatrix
   *
   * @par Overview
   * Constructs a DynamicMatrix from another compatible DynamicMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check. Note that upon
   * construction the new DynamicMatrix will set to have the same active type as
   * the source Matrix.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are constructing from.
   */
  template <class VR, class... PR>
  DynamicMatrix(
      const DynamicMatrix<VR, PR...> &src,
      typename std::enable_if<is_format_compatible<
          DynamicMatrix, DynamicMatrix<VR, PR...>>::value>::type * = nullptr) {
    this->activate(src.active_index());  // switch to src format
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    Morpheus::Impl::Variant::visit(Impl::any_type_assign(), src.const_formats(),
                                   _formats);
  }

  /**
   * @brief Assigns a DynamicMatrix from another compatible DynamicMatrix
   *
   * @par Overview
   * Assigns a DynamicMatrix from another compatible DynamicMatrix i.e a
   * matrix that satisfies the \p is_format_compatible check. Note that upon
   * assignment the new DynamicMatrix will set to have the same active type as
   * the source Matrix.
   *
   * @tparam VR Type of Values the Other Matrix holds.
   * @tparam PR Properties of the Other Matrix.
   * @param src The matrix we are assigning from.
   */
  template <class VR, class... PR>
  typename std::enable_if<
      is_format_compatible<DynamicMatrix, DynamicMatrix<VR, PR...>>::value,
      DynamicMatrix &>::type
  operator=(const DynamicMatrix<VR, PR...> &src) {
    this->activate(src.active_index());  // switch to src format
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    Morpheus::Impl::Variant::visit(Impl::any_type_assign(), src.const_formats(),
                                   _formats);

    return *this;
  }

  // template <typename... Args>
  // inline void resize(const size_type m, const size_type n,
  //                    const size_type nnz, Args &&... args) {
  //   base::resize(m, n, nnz);
  //   auto f = std::bind(Impl::any_type_resize<ValueType, Properties...>(),
  //                      std::placeholders::_1, m, n, nnz,
  //                      std::forward<Args>(args)...);
  //   return Morpheus::Impl::Variant::visit(f, _formats);
  // }

  /**
   * @brief Resizes DynamicMatrix with the shape and number of non-zero entries
   * of the active type of another DynamicMatrix with different parameters.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source DynamicMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline void resize(const DynamicMatrix<VR, PR...> &src) {
    this->activate(src.format_enum());
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    Morpheus::Impl::Variant::visit(Impl::any_type_resize_from_mat(),
                                   src.const_formats(), _formats);
  }

  /**
   * @brief Resizes DynamicMatrix with the shape and number of non-zero entries
   * of another Matrix with possibly a different storage format.
   * \note Upon resizing the new DynamicMatrix will set to have the same active
   * type as the source Matrix.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source Matrix we are resizing from.
   */
  template <typename Matrix>
  inline void resize(
      const Matrix &src,
      typename std::enable_if<is_variant_member_v<Matrix, variant_type>>::type
          * = nullptr) {
    this->activate(src.format_enum());
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    auto f = std::bind(Impl::any_type_resize_from_mat(), std::cref(src),
                       std::placeholders::_1);
    Morpheus::Impl::Variant::visit(f, _formats);
  }

  /**
   * @brief Allocates DynamicMatrix with the shape and number of non-zero
   * entries of the active type of another DynamicMatrix with different
   * parameters.
   * \note Upon allocation the new DynamicMatrix will set to have the same
   * active type as the source Matrix.
   *
   * @tparam VR Type of values the source matrix stores.
   * @tparam PR Other properties of source matrix.
   * @param src The source DynamicMatrix we are resizing from.
   */
  template <class VR, class... PR>
  inline DynamicMatrix &allocate(const DynamicMatrix<VR, PR...> &src) {
    this->activate(src.active_index());  // switch to src format
    base::resize(src.nrows(), src.ncols(), src.nnnz());

    Morpheus::Impl::Variant::visit(Impl::any_type_allocate(),
                                   src.const_formats(), _formats);
    return *this;
  }

  /**
   * @brief Returns the format index assigned to the active type of the
   * DynamicMatrix container.
   *
   * @return formats_e The format enum
   */
  inline int active_index() const { return _formats.index(); }

  /**
   * @brief Returns the format index assigned to the active type of the
   * DynamicMatrix container.
   *
   * @return formats_e The format enum
   */
  int format_index() const { return this->active_index(); }

  /**
   * @brief Returns the format enum assigned to the active type of the
   * DynamicMatrix container.
   *
   * @return formats_e The format enum
   */
  inline formats_e active_enum() const {
    return static_cast<formats_e>(_formats.index());
  }

  /**
   * @brief Returns the format enum assigned to the active type of the
   * DynamicMatrix container.
   *
   * @return formats_e The format enum
   */
  inline formats_e format_enum() const { return this->active_enum(); }

  /**
   * @brief Changes the active type of the DynamicMatrix container to the one
   * given by the enum \p index parameter.
   *
   * @param index The enum state representing the active type to change to.
   */
  inline void activate(const formats_e index) {
    constexpr int size = Morpheus::Impl::Variant::variant_size_v<
        typename MatrixFormats<ValueType, Properties...>::variant>;
    const int idx = static_cast<int>(index);

    if (idx == active_index()) {
      return;
    } else if (idx > size) {
      std::cout << "Warning: There are " << size
                << " available formats to switch to. "
                << "Selecting to switch to format with index " << idx
                << " will default to no change." << std::endl;

      return;
    }

    // Set metadata to 0
    base::resize(0, 0, 0);
    Impl::activate_impl<size, ValueType, Properties...>::activate(_formats,
                                                                  idx);
  }

  /**
   * @brief Changes the active type of the DynamicMatrix container to the one
   * given by the \p index parameter.
   *
   * @param index The index representing the active type to change to.
   */
  inline void activate(const int index) {
    activate(static_cast<formats_e>(index));
  }

  /**
   * @brief Returns a const-reference to the variant container that holds the
   * supported formats in the \p DynamicMatrix.
   *
   * @return const variant_type&  A const-reference to the variant container.
   */
  inline const variant_type &const_formats() const { return _formats; }

  /**
   * @brief Returns a reference to the variant container that holds the
   * supported formats in the \p DynamicMatrix.
   *
   * @return variant_type&  A reference to the variant container.
   */
  inline variant_type &formats() { return _formats; }

 private:
  variant_type _formats;
};
/*! \}  // end of containers_2d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_DYNAMICMATRIX_HPP
