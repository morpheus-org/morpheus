/**
 * concepts.hpp
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

#ifndef MORPHEUS_CORE_CONCEPTS_HPP
#define MORPHEUS_CORE_CONCEPTS_HPP

#include <type_traits>

namespace Morpheus {

// Specify Index Type
template <typename T>
struct IndexType {
  static_assert(std::is_integral<T>::value, "Morpheus: Invalid IndexType<>.");
  using index_type = IndexType;
  using type       = T;
};

// Specify Value Type
template <typename T>
struct ValueType {
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Morpheus: Invalid ValueType<>.");
  using value_type = ValueType;
  using type       = T;
};

//----------------------------------------------------------------------------

#define MORPHEUS_IMPL_IS_CONCEPT(CONCEPT)                                      \
  template <typename T>                                                        \
  struct is_##CONCEPT {                                                        \
   private:                                                                    \
    template <typename, typename = std::true_type>                             \
    struct have : std::false_type {};                                          \
                                                                               \
    template <typename U>                                                      \
    struct have<U, typename std::is_base_of<typename U::CONCEPT, U>::type>     \
        : std::true_type {};                                                   \
                                                                               \
    template <typename U>                                                      \
    struct have<U,                                                             \
                typename std::is_base_of<typename U::CONCEPT##_type, U>::type> \
        : std::true_type {};                                                   \
                                                                               \
   public:                                                                     \
    static constexpr bool value =                                              \
        is_##CONCEPT::template have<typename std::remove_cv<T>::type>::value;  \
                                                                               \
    constexpr operator bool() const noexcept { return value; }                 \
  };

MORPHEUS_IMPL_IS_CONCEPT(index_type)
MORPHEUS_IMPL_IS_CONCEPT(value_type)

MORPHEUS_IMPL_IS_CONCEPT(format_type)
MORPHEUS_IMPL_IS_CONCEPT(memory_space)

#undef MORPHEUS_IMPL_IS_CONCEPT
//----------------------------------------------------------------------------

template <typename T>
struct is_variant : std::false_type {};

template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_variant_v = is_variant<T>::value;

}  // namespace Morpheus

#endif  // MORPHEUS_CORE_CONCEPTS_HPP