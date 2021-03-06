/**
 * matrix_traits.hpp
 * 
 * Edinburgh Parallel Computing Centre (EPCC)
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

#ifndef MORPHEUS_CORE_MATRIX_TRAITS_HPP
#define MORPHEUS_CORE_MATRIX_TRAITS_HPP

#include <morpheus/core/concepts.hpp>
#include <morpheus/core/matrix_tags.hpp>

namespace Morpheus{
    
    // Format Wrapper Type
    template <class T>
    struct FormatType {
        static_assert(std::is_base_of<Morpheus::Impl::MatrixTag, T>::value,
                    "Morpheus: Invalid Format<> type.");
        using format_type = FormatType;
        using type        = T;
    };

    namespace Impl
    {
        template <typename IndexType = void, typename ValueType = void,
                  typename FormatType = void, typename MemorySpace = void>
        struct MatrixTraitsBase{
            using type = MatrixTraitsBase<IndexType, ValueType, FormatType, MemorySpace>;
            using index_type = IndexType;
            using value_type = ValueType;
            using memory_space = MemorySpace;
            using format_type = FormatType;
        };

        template <typename MatrixBase, typename IndexType>
        struct SetIndexType {
            static_assert(std::is_void<typename MatrixBase::index_type>::value,
                            "Morpheus Error: More than one index types given");
            using type =
                MatrixTraitsBase<IndexType, 
                                typename MatrixBase::value_type,
                                typename MatrixBase::memory_space,
                                typename MatrixBase::format_type>;
        };

        template <typename MatrixBase, typename ValueType>
        struct SetValueType {
            static_assert(std::is_void<typename MatrixBase::value_type>::value,
                            "Morpheus Error: More than one value types given");
            using type =
                MatrixTraitsBase<typename MatrixBase::index_type, 
                                ValueType,
                                typename MatrixBase::memory_space,
                                typename MatrixBase::format_type>;
        };

        template <typename MatrixBase, typename FormatType>
        struct SetFormatType {
            static_assert(std::is_void<typename MatrixBase::format_type>::value,
                            "Morpheus Error: More than one formats given");
            using type =
                MatrixTraitsBase<typename MatrixBase::index_type, 
                                typename MatrixBase::value_type,
                                FormatType,
                                typename MatrixBase::memory_space>;
        };

        template <typename MatrixBase, typename MemorySpace>
        struct SetMemorySpace {
            static_assert(std::is_void<typename MatrixBase::memory_space>::value,
                            "Morpheus Error: More than one memory spaces given");
            using type =
                MatrixTraitsBase<typename MatrixBase::index_type, 
                                typename MatrixBase::value_type,
                                typename MatrixBase::format_type,
                                MemorySpace>;
        };

        template <typename Base, typename... Traits>
        struct AnalyzeMatrix;

        // FIXME: Should work when both index and value types are ints
        template <typename Base, typename T, typename... Traits>
        struct AnalyzeMatrix<Base, T, Traits...>
            : public AnalyzeMatrix<
                typename std::conditional_t<is_index_type<T>::value, 
                            SetIndexType<Base,T>, 
                            std::conditional_t<std::is_integral<T>::value, 
                                SetIndexType<Base,IndexType<T>>,
                                std::conditional_t<is_value_type<T>::value,
                                    SetValueType<Base,T>,
                                    std::conditional_t<std::is_floating_point<T>::value || std::is_integral<T>::value,
                                        SetValueType<Base, ValueType<T>>,
                                        std::conditional_t<is_format_type<T>::value,
                                            SetFormatType<Base,T>,
                                            std::conditional_t<is_memory_space<T>::value,
                                                SetMemorySpace<Base,T>,
                                                Base>>>>>>::type, 
                Traits...> {};

        template <typename Base>
        struct AnalyzeMatrix<Base> 
        {
            // static constexpr auto execution_space_is_defaulted =
            //     std::is_void<typename Base::execution_space>::value;

            using index_type = 
                typename std::conditional<std::is_void<typename Base::index_type>::value,
                                            IndexType<int>,
                                            // nasty hack to make index_type into an integral_type
                                            // instead of the wrapped IndexType<T> for backwards compatibility
                                            typename Base::index_type>::type::type;

            using value_type =
                typename std::conditional<std::is_void<typename Base::value_type>::value,
                                            ValueType<double>,
                                            // nasty hack to make value_type into an integral_type
                                            // instead of the wrapped ValueType<T> for backwards compatibility
                                            typename Base::value_type>::type::type;
            

            using memory_space = typename Base::memory_space;
            // TODO
            // using memory_space = 
            //     typename std::conditional<is_void<typename Base::memory_space>::value,
            //                                 DefaultMemorySpace,
            //                                 typename Base::memory_space>::type;
            
            // using format_type = typename Base::format_type;
            // TODO
            using format_type = 
                typename std::conditional<std::is_void<typename Base::format_type>::value,
                                            FormatType<CooFormat>::format_type,
                                            typename Base::format_type>::type;
            // static_assert(std::is_void<typename Base::format_type>::value,
            //                 "Morpheus Error: Matrix format_type is not specified.");
            // using format_type = typename Base::format_type;

            using type =
                MatrixTraitsBase<index_type, value_type, format_type, memory_space>;
        };

        template <typename... Traits>
        struct MatrixTraits
            : AnalyzeMatrix<MatrixTraitsBase<>, Traits...>::type {
            using base_t = typename AnalyzeMatrix<MatrixTraitsBase<>, Traits...>::type;
            
            template <class... Args>
            MatrixTraits(MatrixTraits<Args...> const &p) : base_t(p) {}
            MatrixTraits() = default;
        };
    }
}
#endif  //MORPHEUS_CORE_MATRIX_TRAITS_HPP