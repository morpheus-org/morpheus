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

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_Copy.hpp>

#include <fwd/Morpheus_Fwd_DenseVector.hpp>
#include <impl/Morpheus_VectorTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <string>

namespace Morpheus {

template <class ValueType, class... Properties>
class DenseVector : public Impl::VectorTraits<ValueType, Properties...> {
 public:
  using type   = DenseVector<ValueType, Properties...>;
  using traits = Impl::VectorTraits<ValueType, Properties...>;
  using tag    = typename VectorFormatTag<Morpheus::DenseVectorTag>::tag;

  using value_type      = typename traits::value_type;
  using size_type       = size_t;
  using index_type      = int;
  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using HostMirror = DenseVector<
      typename traits::non_const_value_type,
      Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                     typename traits::host_mirror_space::memory_space>>;

  using host_mirror_type = DenseVector<typename traits::non_const_value_type,
                                       typename traits::host_mirror_space>;

  using pointer         = DenseVector*;
  using const_pointer   = const DenseVector*;
  using reference       = DenseVector&;
  using const_reference = const DenseVector&;

  using value_array_type      = Kokkos::View<value_type*, device_type>;
  using value_array_pointer   = typename value_array_type::pointer_type;
  using value_array_reference = typename value_array_type::reference_type;

  // static_assert(
  //     std::is_same<
  //         typename value_array_type::execution_space,
  //         typename Kokkos::DefaultHostExecutionSpace::execution_space>::value
  //         ||
  //     std::is_same<typename value_array_type::execution_space,
  //                  typename Kokkos::Serial::execution_space>::value);
  // static_assert(
  //     std::is_same<
  //         typename value_array_type::memory_space,
  //         typename Kokkos::DefaultHostExecutionSpace::memory_space>::value ||
  //     std::is_same<typename value_array_type::memory_space,
  //                  typename Kokkos::Serial::memory_space>::value);

 public:
  //   Member functions
  ~DenseVector()                  = default;
  DenseVector(const DenseVector&) = default;
  DenseVector(DenseVector&&)      = default;
  reference operator=(const DenseVector&) = default;
  reference operator=(DenseVector&&) = default;

  inline DenseVector() : _name("Vector"), _size(0), _values() {}

  inline DenseVector(const std::string name, index_type n, value_type val = 0)
      : _name(name + "Vector"), _size(n), _values(name, size_t(n)) {
    std::cout << "Before assigning" << std::endl;
    assign(n, val);
    std::cout << "After assigning" << std::endl;
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

  inline void assign(const index_type n, const value_type val) {
    /* Resize if necessary (behavior of std:vector) */
    // this->resize(n);

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