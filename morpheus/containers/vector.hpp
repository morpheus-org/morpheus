/**
 * dense_vector.hpp
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
#ifndef MORPHEUS_CONTAINERS_DENSE_VECTOR_HPP
#define MORPHEUS_CONTAINERS_DENSE_VECTOR_HPP

#include <morpheus/core/core.hpp>
#include <morpheus/core/vector_traits.hpp>
#include <morpheus/core/vector_tags.hpp>

namespace Morpheus {

/* Forward declaration */
template <class... Properties>
class DenseVector;
/* Alias type to match std::vector */
template <class... Properties>
using vector = DenseVector<Properties...>;

template <class... Properties>
class DenseVector : public Impl::VectorTraits<Properties...> {
 public:
  using type   = DenseVector<Properties...>;
  using traits = Impl::VectorTraits<Properties...>;
  using tag    = typename VectorFormatTag<Impl::DenseVectorTag>::tag;

  using value_type      = typename traits::value_type;
  using memory_space    = typename traits::memory_space;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;

  using pointer         = value_type*;
  using const_pointer   = const value_type*;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using iterator        = value_type*;
  using const_iterator  = const value_type*;
  using size_type       = size_t;  // should that be IndexType as in matrices ?

  using array_type = Kokkos::View<pointer, Kokkos::LayoutRight, memory_space>;

 public:
  //   Member functions
  ~DenseVector()                  = default;
  DenseVector(const DenseVector&) = default;
  DenseVector(DenseVector&&)      = default;
  DenseVector& operator=(const DenseVector&) = default;
  DenseVector& operator=(DenseVector&&) = default;

  inline DenseVector() : _data() {
    _size          = 0;
    _extra_storage = 1.1;
  }

  inline DenseVector(int n, value_type val = 0)
      : _size(n), _extra_storage(1.1), _data("Vector", size_t(n * 1.1)) {
    assign(n, val);
  }

  inline void assign(size_t n, const_reference val) {
    /* Resize if necessary (behavior of std:vector) */

    if (n > _data.span()) Kokkos::resize(_data, size_t(n * _extra_storage));
    _size = n;

    Kokkos::RangePolicy<execution_space, size_type> range(0, n);
    Kokkos::parallel_for(
        "Morpheus::DenseVector::assign", range,
        KOKKOS_LAMBDA(const int i) { _data(i) = val; });
  }

  // Element access
  inline reference operator()(int i) const { return _data(i); }

  inline reference operator[](int i) const { return _data(i); }

  inline size_type size() const { return _size; }

  inline pointer data() const { return _data.data(); }

  // Iterators
  inline iterator begin() { return _data.data(); }

  inline iterator end() {
    return _size > 0 ? _data.data() + _size : _data.data();
  }

  inline const_iterator cbegin() const { return _data.data(); }

  inline const_iterator cend() const {
    return _size > 0 ? _data.data() + _size : _data.data();
  }

  // Capacity
  //   TODO: reserve should be enabled when push_back methods etc are developed
  //   inline void reserve(size_t n) {
  //     Kokkos::resize(_data, size_t(n * _extra_storage));
  //   }

  // Modifiers
  inline void resize(size_t n) {
    if (n > _data.span()) Kokkos::resize(_data, size_t(n * _extra_storage));
    _size = n;
  }

  inline void resize(size_t n, const_reference val) { assign(n, val); }

  // TODO: Data management routines for copying to and from a space

 private:
  size_t _size;
  float _extra_storage;
  array_type _data;
};
}  // namespace Morpheus

#endif  // MORPHEUS_CONTAINERS_DENSE_VECTOR_HPP