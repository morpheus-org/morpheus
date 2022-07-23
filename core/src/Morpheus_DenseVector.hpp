/**
 * Morpheus_DenseVector.hpp
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
#ifndef MORPHEUS_DENSEVECTOR_HPP
#define MORPHEUS_DENSEVECTOR_HPP

#include <Morpheus_TypeTraits.hpp>
#include <Morpheus_FormatTags.hpp>

#include <impl/Morpheus_ContainerTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace Morpheus {

/**
 * \addtogroup containers Containers
 * \par Overview
 * TODO
 *
 */

namespace Impl {
// TODO: Merge this and set_functor from DenseMatrix in a single definition and
// place in Impl directory
template <typename View, typename ValueType>
struct set_functor {
  View _data;
  ValueType _val;

  set_functor(View data, ValueType val) : _data(data), _val(val) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const { _data(i) = _val; }
};
}  // namespace Impl
/**
 * \addtogroup containers_1d 1D Containers
 * \brief One-dimensional Containers
 * \ingroup containers
 * \{
 *
 */

/**
 * @brief The DenseVector container is a one-dimensional container that contains
 * contiguous elements. It is a polymorphic container in the sense that it can
 * store scalar or integer type values, on host or device depending how the
 * template parameters are selected.
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
 *  Morpheus::DenseVector<double, Kokkos::HostSpace> x(10, 5.0);
 *
 *  // Set some values
 *  x[2] = 5.0;
 *  x[5] = -2.0;
 * }
 * \endcode
 */
template <class ValueType, class... Properties>
class DenseVector
    : public Impl::ContainerTraits<DenseVector, ValueType, Properties...> {
 public:
  /*! The traits associated with the particular container */
  using traits = Impl::ContainerTraits<DenseVector, ValueType, Properties...>;
  /*! The complete type of the container */
  using type = typename traits::type;
  /*! The tag associated specificaly to the particular container*/
  using tag = typename VectorFormatTag<Morpheus::DenseVectorFormatTag>::tag;

  /*! The type of the values held by the container - can be const */
  using value_type = typename traits::value_type;
  /*! The non-constant type of the values held by the container */
  using non_const_value_type = typename traits::non_const_value_type;
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

  /*! The type of view that holds the value_type data */
  using value_array_type =
      Kokkos::View<value_type*, array_layout, execution_space, memory_traits>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

  //   Member functions
  ~DenseVector() = default;
  /**
   * @brief Default copy contructor (shallow copy) of a DenseVector container
   * from another DenseVector container with the same properties.
   */
  DenseVector(const DenseVector&) = default;
  /**
   * @brief Default move contructor (shallow copy) of a DenseVector container
   * from another DenseVector container with the same properties.
   */
  DenseVector(DenseVector&&) = default;
  /**
   * @brief Default copy assignment (shallow copy) of a DenseVector container
   * from another DenseVector container with the same properties.
   */
  DenseVector& operator=(const DenseVector&) = default;
  /**
   * @brief Default move assignment (shallow copy) of a DenseVector container
   * from another DenseVector container with the same properties.
   */
  DenseVector& operator=(DenseVector&&) = default;

  /**
   * @brief Construct an empty DenseVector object
   */
  inline DenseVector() : _size(0), _values() {}

  /**
   * @brief Construct a DenseVector object with size \p n and values set to
   * \p val
   *
   * @param n Size of the DenseVector
   * @param val Value at which the elements of the DenseVector will be set to
   */
  inline DenseVector(const size_t n, value_type val = 0)
      : _size(n), _values("vector", size_t(n)) {
    assign(n, val);
  }

  /**
   * @brief Construct a DenseVector object from a raw pointer. This is only
   * enabled if the DenseVector is an unmanaged container.
   *
   * @tparam ValuePtr Pointer type
   * @param n Number of entries
   * @param ptr Pointer value
   */
  template <typename ValuePtr>
  explicit DenseVector(
      const size_t n, ValuePtr ptr,
      typename std::enable_if<std::is_pointer<ValuePtr>::value &&
                              (memory_traits::is_unmanaged)>::type* = nullptr)
      : _size(n), _values(ptr, n) {
    static_assert(std::is_same<value_array_pointer, ValuePtr>::value,
                  "Constructing DenseVector to wrap user memory must supply "
                  "matching pointer type");
  }

  /**
   * @brief Construct a DenseVector object with values from
   * \p range_low to \p range_high
   *
   * @tparam Generator random number generator type
   * @param n Size of DenseVector
   * @param rand_pool Random number generator
   * @param range_low Low bound value to assign - included
   * @param range_high Upper bound value to assign - excluded
   */
  template <typename Generator>
  inline DenseVector(const size_t n, Generator rand_pool,
                     const value_type range_low, const value_type range_high)
      : _size(n), _values("vector", n) {
    Kokkos::fill_random(_values, rand_pool, range_low, range_high);
  }

  /**
   * @brief Shallow Copy contrustor from another DenseVector container with
   * different properties. Note that this is only possible when the \p
   * is_compatible check is satisfied.
   *
   * @tparam VR Value Type of the container we are constructing from.
   * @tparam PR Optional properties of the container we are constructing from.
   * @param src The \p DenseVector container we are constructing from.
   */
  template <class VR, class... PR>
  inline DenseVector(
      const DenseVector<VR, PR...>& src,
      typename std::enable_if_t<
          is_compatible<DenseVector, DenseVector<VR, PR...>>::value>* = nullptr)
      : _size(src.size()), _values(src.const_view()) {}

  /**
   * @brief Shallow Copy Assignemnt from another DenseVector container with
   * different properties. Note that this is only possible when the \p
   * is_compatible check is satisfied.
   *
   * @tparam VR Value Type of the container we are copying from.
   * @tparam PR Optional properties of the container we are copying from.
   * @param src The \p DenseVector container we are copying from.
   */
  template <class VR, class... PR>
  typename std::enable_if_t<
      is_compatible<DenseVector, DenseVector<VR, PR...>>::value, DenseVector&>
  operator=(const DenseVector<VR, PR...>& src) {
    _size   = src.size();
    _values = src.const_view();
    return *this;
  }

  /**
   * @brief Allocates memory from another DenseVector container with
   * different properties.
   *
   * @tparam VR Value Type of the container we are allocating from.
   * @tparam PR Optional properties of the container we are allocating from.
   * @param src The \p DenseVector container we are allocating from.
   */
  template <class VR, class... PR>
  inline DenseVector& allocate(const DenseVector<VR, PR...>& src) {
    _size = src.size();
    this->resize(src.size());
    return *this;
  }

  /**
   * @brief Assigns \p n elements of value \p val to the DenseVector. In the
   * case where \p n is larger than the actual size of the container, the
   * container will resize before assignment.
   *
   * @param n Number of elements to assign
   * @param val Value to assign
   */
  inline void assign(const size_t n, const value_type val) {
    using range_policy = Kokkos::RangePolicy<size_t, execution_space>;

    /* Resize if necessary (behavior of std:vector) */
    if (n > _size) {
      this->resize(n);
    }

    range_policy policy(0, n);
    Impl::set_functor f(_values, val);
    Kokkos::parallel_for("Morpheus::DenseVector::assign", policy, f);
  }

  /**
   * @brief Assigns \p n elements of values between \p range_low  and
   * \p range_high to the DenseVector. In the case where \p n is larger than the
   * actual size of the container, the container will resize before assignment.
   *
   * @tparam Generator random number generator type
   * @param n Number of elements to assign
   * @param rand_pool Random number generator
   * @param range_low Low bound value to assign
   * @param range_high Upper bound value to assign
   */
  template <typename Generator>
  inline void assign(const size_t n, Generator rand_pool,
                     const value_type range_low, const value_type range_high) {
    /* Resize if necessary (behavior of std:vector) */
    if (n > _size) {
      this->resize(n);
    }
    auto vals = Kokkos::subview(_values, std::make_pair((size_t)0, n));
    Kokkos::fill_random(vals, rand_pool, range_low, range_high);
  }

  /**
   * @brief Returns a reference to the element with index \p i
   *
   * @param i Index of the value to extract
   * @return Element at index \p i
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference
  operator()(const size_t i) const {
    return _values(i);
  }

  /**
   * @brief Returns a reference to the element with index \p i
   *
   * @param i Index of the value to extract
   * @return Element at index \p i
   */
  MORPHEUS_FORCEINLINE_FUNCTION value_array_reference
  operator[](const size_t i) const {
    return _values(i);
  }

  /**
   * @brief Returns the size of the container
   *
   * @return Integer representing the size of the container
   */
  inline size_t size() const { return _size; }

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
  inline value_array_type& view() { return _values; }

  /**
   * @brief Returns a constant reference to the beginning of the view that holds
   * the data
   *
   * @return Constant type of view that holds the data
   */
  inline const value_array_type& const_view() const { return _values; }

  /**
   * @brief Resizes DenseVector with size of \p n and sets values to zero.
   *
   * @param n New size of the container
   */
  inline void resize(const size_t n) {
    Kokkos::resize(_values, size_t(n));
    _size = n;
  }

  /**
   * @brief Resizes DenseVector with size of \p n and sets values to \p val.
   * Note that compared to \p assign() member function, resize operation always
   * changes the size of the container.
   *
   * @param n New size of the container
   */
  inline void resize(const size_t n, const value_type val) {
    resize(n);
    assign(n, val);
  }

 private:
  size_t _size;
  value_array_type _values;
};

/*! \}  // end of containers_1d group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_HPP