if(NOT CUDAToolkit_ROOT)
  if(NOT CUDA_ROOT)
    set(CUDA_ROOT $ENV{CUDA_ROOT})
  endif()
  if(CUDA_ROOT)
    set(CUDAToolkit_ROOT ${CUDA_ROOT})
  endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/CudaToolkit.cmake)

include(FindPackageHandleStandardArgs)

if(TARGET CUDA::cublas)
  set(FOUND_CUBLAS TRUE)
  morpheus_export_imported_tpl(CUDA::cublas IMPORTED_NAME CUDA::cublas)
else()
  set(FOUND_CUBLAS FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TPLCUBLAS "CuBLAS and cuBLAS_LT were not found!" FOUND_CUBLAS)
if(FOUND_CUBLAS)
  morpheus_create_imported_tpl(CUBLAS INTERFACE LINK_LIBRARIES CUDA::cublas)
endif()
