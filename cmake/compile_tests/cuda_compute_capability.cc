/**
 * cuda_compute_capability.cc
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

#include <iostream>

int main() {
  cudaDeviceProp device_properties;
  const cudaError_t error = cudaGetDeviceProperties(&device_properties,
                                                    /*device*/ 0);
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << '\n';
    return error;
  }
  unsigned int const compute_capability =
      device_properties.major * 10 + device_properties.minor;
#ifdef SM_ONLY
  std::cout << compute_capability;
#else
  switch (compute_capability) {
      // clang-format off
    case 60: std::cout << "Set -DMorpheus_ARCH_PASCAL60=ON ." << std::endl; break;
    case 61: std::cout << "Set -DMorpheus_ARCH_PASCAL61=ON ." << std::endl; break;
    case 70: std::cout << "Set -DMorpheus_ARCH_VOLTA70=ON ." << std::endl; break;
    case 72: std::cout << "Set -DMorpheus_ARCH_VOLTA72=ON ." << std::endl; break;
    case 75: std::cout << "Set -DMorpheus_ARCH_TURING75=ON ." << std::endl; break;
    case 80: std::cout << "Set -DMorpheus_ARCH_AMPERE80=ON ." << std::endl; break;
    case 86: std::cout << "Set -DMorpheus_ARCH_AMPERE86=ON ." << std::endl; break;
    default:
      std::cout << "Compute capability " << compute_capability
                << " is not supported" << std::endl;
      // clang-format on
  }
#endif
  return 0;
}