message(STATUS "Core Benchmarks are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

morpheus_add_executable_and_test(Benchmarks_Spmv_Serial SOURCES
                                 Benchmarks_Spmv.cpp)
target_compile_definitions(MorpheusCore_Benchmarks_Spmv_Serial
                           PUBLIC WITH_SERIAL)

if(MORPHEUS_ENABLE_OPENMP)
  morpheus_add_executable_and_test(Benchmarks_Spmv_OpenMP SOURCES
                                   Benchmarks_Spmv.cpp)
  target_compile_definitions(MorpheusCore_Benchmarks_Spmv_OpenMP
                             PUBLIC WITH_OPENMP)
endif()

if(MORPHEUS_ENABLE_CUDA)
  morpheus_add_executable_and_test(Benchmarks_Spmv_Cuda SOURCES
                                   Benchmarks_Spmv.cpp)
  target_compile_definitions(MorpheusCore_Benchmarks_Spmv_Cuda PUBLIC WITH_CUDA)
endif()

morpheus_add_executable_and_test(Benchmarks_Copy SOURCES Benchmarks_Copy.cpp)

morpheus_add_executable_and_test(Benchmarks_Convert SOURCES
                                 Benchmarks_Convert.cpp)
