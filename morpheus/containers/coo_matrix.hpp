/**
 * coo_matrix.hpp
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

#ifndef MORPHEUS_CONTAINERS_COO_MATRIX_HPP
#define MORPHEUS_CONTAINERS_COO_MATRIX_HPP

#include <vector>
#include <string>
#include <iostream>

#include <morpheus/core/matrix_traits.hpp>
#include <morpheus/core/exceptions.hpp>

namespace Morpheus
{
    namespace Impl
    {
        struct CooFormat : public MatrixTag {};
    }

    template<class... Properties>
    class CooMatrix : public Impl::MatrixTraits<FormatType<Impl::CooFormat>, Properties...>
    {
    public:
        using type = CooMatrix<Properties...>;
        using traits = Impl::MatrixTraits<FormatType<Impl::CooFormat>, Properties...>;
        using index_type = typename traits::index_type;
        using value_type = typename traits::value_type;
        using format_type = typename traits::format_type;

        // TODO: Use Morpheus::array instead of std::vector
        using index_array_type = std::vector<index_type>;
        using value_array_type = std::vector<value_type>;

        // TODO: Make arrays private
        index_array_type row_indices, column_indices;
        value_array_type values;

        // Construct an empty CooMatrix
        inline CooMatrix(){}

        // Construct a CooMatrix with a specific shape and number of non-zero entries
        inline CooMatrix(const index_type num_rows, const index_type num_cols, const index_type num_entries)
            : _m(num_rows), _n(num_cols), _nnz(num_entries),
              row_indices(num_entries), 
              column_indices(num_entries), 
              values(num_entries){}

        // Construct from another matrix type
        template <typename MatrixType>
        CooMatrix(const MatrixType& matrix)
        { 
            // TODO: CooMatrix(const MatrixType& matrix)
            Morpheus::NotImplementedException("CooMatrix(const MatrixType& matrix)");
        }

        // Resize matrix dimensions and underlying storage
        inline void resize(const index_type num_rows, const index_type num_cols, const index_type num_entries)
        {   
            _m = num_rows;
            _n = num_cols;
            _nnz = num_entries;
            row_indices.resize(_nnz);
            column_indices.resize(_nnz);
            values.resize(_nnz);
        }

        // Swap the contents of two CooMatrix objects.
        void swap(CooMatrix& matrix)
        { 
            // TODO: CooMatrix.swap
            Morpheus::NotImplementedException("CooMatrix.swap(const MatrixType& matrix)"); 
        }

        // Assignment from another matrix type
        template <typename MatrixType>
        CooMatrix& operator=(const MatrixType& matrix)
        { std::cout << "CooMatrix.operator=(const MatrixType& matrix)" << std::endl; }

        // Sort matrix elements by row index
        void sort_by_row(void)
        { 
            // TODO: CooMatrix.sort_by_row
            Morpheus::NotImplementedException("CooMatrix.sort_by_row()"); 
        }

        // Sort matrix elements by row and column index
        void sort_by_row_and_column(void)
        { 
            // TODO: CooMatrix.sort_by_row_and_column
            Morpheus::NotImplementedException("CooMatrix.sort_by_row_and_column()"); 
        }

        // Determine whether matrix elements are sorted by row index
        bool is_sorted_by_row(void)
        { 
            // TODO: CooMatrix.is_sorted_by_row
            Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row()"); 
        }

        // Determine whether matrix elements are sorted by row and column index
        bool is_sorted_by_row_and_column(void)
        { 
            // TODO: CooMatrix.is_sorted_by_row_and_column
            Morpheus::NotImplementedException("CooMatrix.is_sorted_by_row_and_column()"); 
        }
        
        inline std::string name() { return _name; }
        inline index_type nrows() { return _m; }
        inline index_type ncols() { return _n; }
        inline index_type nnnz()  { return _nnz; }

    private:
        std::string _name = "CooMatrix";
        index_type _m, _n, _nnz;
    };
}

#endif  //MORPHEUS_CONTAINERS_COO_MATRIX_HPP