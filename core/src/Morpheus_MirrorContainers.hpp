/**
 * Morpheus_MirrorContainers.hpp
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

#ifndef MORPHEUS_MIRRORCONTAINERS_HPP
#define MORPHEUS_MIRRORCONTAINERS_HPP

#include <Morpheus_DenseVector.hpp>
#include <Morpheus_TypeTraits.hpp>

namespace Morpheus {

namespace Impl {
template <class Space, template <class, class...> class Container, class T,
          class... P>
struct MirrorContainerType {
  // The incoming container_type
  using src_container_type = Container<T, P...>;
  // The memory space for the mirror container
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space,
                     typename src_container_type::memory_space>::value
  };
  // The array_layout
  using array_layout = typename src_container_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using value_type = typename src_container_type::non_const_value_type;
  using index_type = typename src_container_type::index_type;
  // The destination container type if it is not the same memory space
  using dest_container_type =
      Container<value_type, index_type, array_layout, Space>;
  // If it is the same memory_space return the existsing container_type
  using container_type =
      typename std::conditional<is_same_memspace, src_container_type,
                                dest_container_type>::type;
};

template <class Space, template <class, class...> class Container, class T,
          class... P>
struct MirrorType {
  // The incoming container_type
  using src_container_type = Container<T, P...>;
  // The memory space for the mirror container
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space,
                     typename src_container_type::memory_space>::value
  };
  // The array_layout
  using array_layout = typename src_container_type::array_layout;
  // we want it non-const to allow deep_copy.
  using value_type = typename src_container_type::non_const_value_type;
  using index_type = typename src_container_type::non_const_index_type;
  // The destination container type if it is not the same memory space
  using container_type = Container<value_type, index_type, array_layout, Space>;
};
}  // namespace Impl

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror(
    const Container<T, P...>& src,
    typename std::enable_if<
        is_sparse_matrix_class<Container, T, P...>::value>::type* = nullptr) {
  using src_type = Container<T, P...>;
  using dst_type = typename src_type::HostMirror;

  return dst_type(src.name().append("Mirror_"), src.nrows(), src.ncols(),
                  src.nnnz());
}

// Allocates a mirror vector with the same characteristics as source
// Doesn't copy elements from source to mirror
template <class T, class... P>
typename DenseVector<T, P...>::HostMirror create_mirror(
    const DenseVector<T, P...>& src) {
  using src_type = DenseVector<T, P...>;
  using dst_type = typename src_type::HostMirror;

  return dst_type(src.name().append("Mirror_"), src.size());
}

// Create a mirror in a new space (specialization for different space)
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorType<Space, Container, T, P...>::container_type
create_mirror(
    const Container<T, P...>& src,
    typename std::enable_if<
        is_sparse_matrix_class<Container, T, P...>::value>::type* = nullptr) {
  return typename Impl::MirrorType<Space, Container, T, P...>::container_type(
      src.name().append("Mirror_"), src.nrows(), src.ncols(), src.nnnz());
}

// Allocates a mirror vector with the same characteristics as source but in new
// Space. Doesn't copy elements from source to mirror
template <class Space, class T, class... P>
typename Impl::MirrorType<Space, DenseVector, T, P...>::container_type
create_mirror(const DenseVector<T, P...>& src) {
  return typename Impl::MirrorType<Space, DenseVector, T, P...>::container_type(
      src.name().append("Mirror"), src.size());
}

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<
        (std::is_same<
             typename Container<T, P...>::memory_space,
             typename Container<T, P...>::HostMirror::memory_space>::value &&
         std::is_same<
             typename Container<T, P...>::value_type,
             typename Container<T, P...>::HostMirror::value_type>::value &&
         std::is_same<typename Container<T, P...>::array_layout,
                      typename Container<T, P...>::HostMirror::array_layout>::
             value)>::type* = nullptr) {
  return src;
}

template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<
        !(std::is_same<
              typename Container<T, P...>::memory_space,
              typename Container<T, P...>::HostMirror::memory_space>::value &&
          std::is_same<
              typename Container<T, P...>::value_type,
              typename Container<T, P...>::HostMirror::value_type>::value &&
          std::is_same<typename Container<T, P...>::array_layout,
                       typename Container<T, P...>::HostMirror::array_layout>::
              value)>::type* = nullptr) {
  return Morpheus::create_mirror(src);
}

// Create a mirror container in a new space (specialization for same space)
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorContainerType<Space, Container, T, P...>::container_type
create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<Impl::MirrorContainerType<
        Space, Container, T, P...>::is_same_memspace>::type* = nullptr) {
  return src;
}

// Create a mirror container in a new space (specialization for different space)
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorContainerType<Space, Container, T, P...>::container_type
create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<
        !Impl::MirrorContainerType<Space, Container, T,
                                   P...>::is_same_memspace &&
        is_sparse_matrix<Container<T, P...>>::value>::type* = nullptr) {
  return typename Impl::MirrorContainerType<Space, Container, T, P...>::
      container_type(src.name().append("MirrorContainer_"), src.nrows(),
                     src.ncols(), src.nnnz());
}

// Create a mirror DenseVector in a new space (specialization for different
// space)
template <class Space, class T, class... P>
typename Impl::MirrorContainerType<Space, DenseVector, T, P...>::container_type
create_mirror_container(
    const DenseVector<T, P...>& src,
    typename std::enable_if<!Impl::MirrorContainerType<
        Space, DenseVector, T, P...>::is_same_memspace>::type* = nullptr) {
  return typename Impl::MirrorContainerType<Space, DenseVector, T, P...>::
      container_type(src.name().append("MirrorContainer_"), src.size());
}

}  // namespace Morpheus

#endif  // MORPHEUS_MIRRORCONTAINERS_HPP