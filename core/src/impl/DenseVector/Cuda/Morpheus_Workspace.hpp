/**
 * Morpheus_Workspace_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_CUDA_WORKSPACE_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_CUDA_WORKSPACE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_CUDA)

#include <Morpheus_FormatTags.hpp>
#include <Morpheus_TypeTraits.hpp>

#include <impl/Morpheus_CudaUtils.hpp>

namespace Morpheus {
namespace Impl {

class CudaWorkspace {
 public:
  CudaWorkspace() : _workspace(nullptr), _nbytes(0) {}

  ~CudaWorkspace() { _free(); }

  template <typename ValueType>
  void allocate(size_t N) {
    size_t bytes = N * sizeof(ValueType);
    if ((bytes > _nbytes) && (_nbytes > 0)) {
      _free();
    }
    _malloc(bytes);
  }

  template <typename ValueType>
  ValueType* data() {
    return reinterpret_cast<ValueType*>(_workspace);
  }

 private:
  void* _workspace;
  size_t _nbytes;

  void _malloc(size_t bytes) {
    if (_workspace == nullptr) {
      _nbytes = bytes;
      if (cudaSuccess != cudaMalloc(&_workspace, bytes)) {
        std::cout << "Malloc Failed on Device" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  void _free() {
    if (_workspace != nullptr) {
      if (cudaSuccess == cudaFree(_workspace)) {
        std::cout << "Free Succeeded!" << std::endl;
        _workspace = nullptr;
        _nbytes    = 0;
      }
    }
  }
};

extern CudaWorkspace cudotspace;

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_CUDA
#endif  // MORPHEUS_DENSEVECTOR_CUDA_WORKSPACE_IMPL_HPP