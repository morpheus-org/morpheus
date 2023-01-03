/**
 * Morpheus_MirrorContainers.hpp
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

#ifndef MORPHEUS_MIRRORCONTAINERS_HPP
#define MORPHEUS_MIRRORCONTAINERS_HPP

#include <Morpheus_FormatTraits.hpp>

#include <impl/Morpheus_MirrorContainers_Impl.hpp>

namespace Morpheus {

/**
 * \defgroup data_management Data Management
 * \par Overview
 * TODO
 *
 */

/**
 * \addtogroup mirroring Mirroring
 * \brief Mirroring Operations on Containers
 * \ingroup data_management
 * \{
 *
 */

/**
 * @brief Allocates a mirror with the same characteristics as source on Host
 * (specialization for different space)
 *
 * @tparam Container
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return typename Container<T, P...>::HostMirror A mirror of the original
 * container on Host
 *
 * \note \p create_mirror operation always issues a new allocation and doesn't
 * copy elements from source to mirror.
 */
template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror(
    const Container<T, P...>& src,
    typename std::enable_if_t<is_container_v<Container<T, P...>>>* = nullptr) {
  using src_type = Container<T, P...>;
  using dst_type = typename src_type::HostMirror;

  return dst_type().allocate(src);
}

/**
 * @brief Create a mirror in a new space (specialization for different space)
 *
 * @tparam Space The new space in which the mirror is created
 * @tparam Container
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return Impl::MirrorType<Space, Container, T, P...>::container_type A mirror
 * of the original container in Space
 *
 * \note \p create_mirror operation always issues a new allocation.
 */
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorType<Space, Container, T, P...>::container_type
create_mirror(
    const Container<T, P...>& src,
    typename std::enable_if_t<is_container_v<Container<T, P...>>>* = nullptr) {
  using container_type =
      typename Impl::MirrorType<Space, Container, T, P...>::container_type;
  return container_type().allocate(src);
}

/**
 * @brief Creates a mirror container on Host (specialization for
 * same space)
 *
 * @tparam Container The type of the container to mirror
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return Container<T, P...>::HostMirror The Host Mirror type of source
 * container
 *
 * \note Here no actual mirror is created as the source container already lives
 * on Host. As a result the resulting container aliases the source.
 */
template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if_t<is_compatible_v<
        Container<T, P...>, typename Container<T, P...>::HostMirror>>* =
        nullptr) {
  return src;
}

/**
 * @brief Creates a mirror container on Host (specialization for
 * different space)
 *
 * @tparam Container The type of the container to mirror
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return Container<T, P...>::HostMirror The Host Mirror type of source
 * container
 */
template <template <class, class...> class Container, class T, class... P>
typename Container<T, P...>::HostMirror create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if_t<!is_compatible_v<
        Container<T, P...>, typename Container<T, P...>::HostMirror>>* =
        nullptr) {
  return Morpheus::create_mirror(src);
}

/**
 * @brief Create a mirror container in a new space (specialization for same
 * space)
 *
 * @tparam Space The new space in which the mirror is created
 * @tparam Container The type of the container to mirror
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return Impl::MirrorContainerType<Space, Container, T, P...>::container_type
 * Same type as source
 *
 * \note Here no actual mirror is created as the source container already lives
 * in the same space with resulting container. As a result the resulting
 * container aliases the source.
 */
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorContainerType<Space, Container, T, P...>::container_type
create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<Impl::MirrorContainerType<Space, Container, T,
                                                      P...>::is_same_memspace &&
                            is_container<Container<T, P...>>::value>::type* =
        nullptr) {
  return src;
}

/**
 * @brief Creates a mirror container in a new space (specialization for
 * different space)
 *
 * @tparam Space The new space in which the mirror is created
 * @tparam Container The type of the container to mirror
 * @tparam T The type of values held by the container
 * @tparam P Properties of the container
 * @param src The source container we are mirroring from
 * @return Impl::MirrorContainerType<Space, Container, T, P...>::container_type
 * A mirror of the original container in Space
 */
template <class Space, template <class, class...> class Container, class T,
          class... P>
typename Impl::MirrorContainerType<Space, Container, T, P...>::container_type
create_mirror_container(
    const Container<T, P...>& src,
    typename std::enable_if<!Impl::MirrorContainerType<
                                Space, Container, T, P...>::is_same_memspace &&
                            is_container<Container<T, P...>>::value>::type* =
        nullptr) {
  using container_type =
      typename Impl::MirrorContainerType<Space, Container, T,
                                         P...>::container_type;
  return container_type().allocate(src);
}
/*! \}  // end of mirroring group
 */
}  // namespace Morpheus

#endif  // MORPHEUS_MIRRORCONTAINERS_HPP