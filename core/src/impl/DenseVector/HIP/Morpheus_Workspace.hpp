/**
 * Morpheus_Workspace_Impl.hpp
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

#ifndef MORPHEUS_DENSEVECTOR_HIP_WORKSPACE_IMPL_HPP
#define MORPHEUS_DENSEVECTOR_HIP_WORKSPACE_IMPL_HPP

#include <Morpheus_Macros.hpp>
#if defined(MORPHEUS_ENABLE_HIP)

#include <impl/Morpheus_HIPUtils.hpp>

#ifdef MORPHEUS_ENABLE_TPL_HIPBLAS
#include <hipblas.h>
#endif  // MORPHEUS_ENABLE_TPL_HIPBLAS

namespace Morpheus {
namespace Impl {

class HIPWorkspace {
 public:
  HIPWorkspace() : _workspace(nullptr), _nbytes(0) {}

  ~HIPWorkspace() { _free(); }

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
      if (hipSuccess != hipMalloc(&_workspace, bytes)) {
        std::cout << "Malloc Failed on Device" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  void _free() {
    if (_workspace != nullptr) {
      if (hipSuccess == hipFree(_workspace)) {
        _workspace = nullptr;
        _nbytes    = 0;
      }
    }
  }
};

#ifdef MORPHEUS_ENABLE_TPL_HIPBLAS

class HIPblasWorkspace : public HIPWorkspace {
 public:
  HIPblasWorkspace() : _handle(nullptr) {}

  ~HIPblasWorkspace() {
    if (_handle != nullptr) {
      if (hipblasDestroy(_handle) != HIPBLAS_STATUS_SUCCESS) {
        _handle = nullptr;
      }
    }
  }

  void init() {
    if (_handle == nullptr) {
      if (hipblasCreate(&_handle) != HIPBLAS_STATUS_SUCCESS) {
        printf("HIPBLAS initialization failed\n");
        exit(EXIT_FAILURE);
      }
      hipblasSetPointerMode(_handle, HIPBLAS_POINTER_MODE_DEVICE);
    }
  }

  hipblasHandle_t handle() { return _handle; }

 private:
  hipblasHandle_t _handle;
};

extern HIPblasWorkspace hipblasdotspace;

#endif  // MORPHEUS_ENABLE_TPL_HIPBLAS

extern HIPWorkspace hipdotspace;

}  // namespace Impl
}  // namespace Morpheus

#endif  // MORPHEUS_ENABLE_HIP
#endif  // MORPHEUS_DENSEVECTOR_HIP_WORKSPACE_IMPL_HPP