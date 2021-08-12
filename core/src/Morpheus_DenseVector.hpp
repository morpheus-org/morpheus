/**
 * Morpheus_DenseVector.hpp
 *
 * EPCC, The University of Edinburgh
 *
 * (c) 2021 The University of Edinburgh
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
#include <Morpheus_Copy.hpp>

#include <fwd/Morpheus_Fwd_DenseVector.hpp>
#include <impl/Morpheus_ContainerTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <string>

namespace Morpheus {

template <class ValueType, class... Properties>
class DenseVector
    : public Impl::ContainerTraits<DenseVector, ValueType, Properties...> {
 public:
  using traits = Impl::ContainerTraits<DenseVector, ValueType, Properties...>;
  using type   = typename traits::type;
  using tag    = typename VectorFormatTag<Morpheus::DenseVectorTag>::tag;

  using value_type           = typename traits::value_type;
  using non_const_value_type = typename traits::non_const_value_type;
  using index_type           = typename traits::index_type;
  using non_const_index_type = typename traits::non_const_index_type;

  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using HostMirror       = typename traits::HostMirror;
  using host_mirror_type = typename traits::host_mirror_type;

  using pointer         = typename traits::pointer;
  using const_pointer   = typename traits::const_pointer;
  using reference       = typename traits::reference;
  using const_reference = typename traits::const_reference;

  using value_array_type      = Kokkos::View<value_type*, device_type>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

 public:
  //   Member functions
  ~DenseVector()                  = default;
  DenseVector(const DenseVector&) = default;
  DenseVector(DenseVector&&)      = default;
  DenseVector& operator=(const DenseVector&) = default;
  DenseVector& operator=(DenseVector&&) = default;

  inline DenseVector() : _name("Vector"), _size(0), _values() {}

  inline DenseVector(const std::string name, index_type n, value_type val = 0)
      : _name(name + "Vector"), _size(n), _values(name, size_t(n)) {
    assign(n, val);
  }

  inline DenseVector(index_type n, value_type val = 0)
      : _name("Vector"), _size(n), _values("Vector", size_t(n)) {
    assign(n, val);
  }
  template <typename Generator>
  inline DenseVector(const std::string name, index_type n, Generator rand_pool,
                     const value_type range_low, const value_type range_high)
      : _name(name + "Vector"), _size(n), _values(name + "Vector", size_t(n)) {
    assign(n, rand_pool, range_low, range_high);
  }

  // inline DenseVector(const std::string name, const DenseVector& src)
  //     : _name(name + "Vector"), _size(src.size()), _values(src.view()) {}

  // inline DenseVector(const std::string name, const DenseVector& src,
  // MirrorTag)
  //     : _name(name + "MirrorVector"),
  //       _size(src.size()),
  //       _values(name + "MirrorVector", size_t(src.size())) {}

  // // Construct from another vector type in different memory space
  // // Only allocates the vector
  // template <class VR, class... PR>
  // DenseVector(const std::string name, const DenseVector<VR, PR...>& vec,
  //             typename std::enable_if<is_compatible_from_different_space<
  //                 DenseVector, DenseVector<VR, PR...>>::value>::type* =
  //                 nullptr)
  //     : _name(name + "Vector_AllocOnly"),
  //       _size(vec.size()),
  //       _values(name + "Vector_AllocOnly", size_t(vec.size())) {}

  // Need to make sure two containers are of compatible type for shallow copy
  template <class VR, class... PR>
  inline DenseVector(
      const DenseVector<VR, PR...>& src,
      typename std::enable_if<is_compatible_type<
          DenseVector, DenseVector<VR, PR...>>::value>::type* = nullptr)
      : _name("ShallowVector"), _size(src.size()), _values(src.view()) {}

  template <class VR, class... PR>
  typename std::enable_if<
      is_compatible_type<DenseVector, DenseVector<VR, PR...>>::value,
      DenseVector&>::type
  operator=(const DenseVector<VR, PR...>& src) {
    if (this != &src) {
      _name   = src.name();
      _size   = src.size();
      _values = src.view();
    }
    return *this;
  }

  // Allocates a vector based on the shape of the source vector
  // Needed for Mirror operations
  template <class VR, class... PR>
  inline DenseVector& allocate(const std::string name,
                               const DenseVector<VR, PR...>& src) {
    _name = name + "_Allocated";
    _size = src.size();
    this->resize(src.size());
    return *this;
  }

  inline void assign(const index_type n, const value_type val) {
    /* Resize if necessary (behavior of std:vector) */
    this->resize(n);

    set_functor f(_values, val);
    Kokkos::parallel_for("Morpheus::DenseVector::assign", n, f);
  }

  // Element access
  inline value_array_reference operator()(index_type i) const {
    return _values(i);
  }

  inline value_array_reference operator[](index_type i) const {
    return _values(i);
  }

  inline index_type size() const { return _size; }

  inline value_array_pointer data() const { return _values.data(); }
  inline const value_array_type& view() const { return _values; }

  // Modifiers
  inline void resize(index_type n) {
    Kokkos::resize(_values, size_t(n));
    _size = n;
  }

  inline void resize(const index_type n, const index_type val) {
    assign(n, val);
  }

  // Other
  inline std::string name() const { return _name; }

 private:
  std::string _name;
  index_type _size;
  value_array_type _values;

 public:
  struct set_functor {
    using execution_space = typename traits::execution_space;
    value_array_type _data;
    value_type _val;

    set_functor(value_array_type data, value_type val)
        : _data(data), _val(val) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const index_type& i) const { _data(i) = _val; }
  };
};

}  // namespace Morpheus

#endif  // MORPHEUS_DENSEVECTOR_HPP