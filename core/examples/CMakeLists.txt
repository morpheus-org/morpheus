message(STATUS "Core Examples are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(MORPHEUS_ENABLE_CUDA)
  morpheus_add_executable_and_test(Examples_Cuda SOURCES Examples_Cuda.cpp)
endif()

morpheus_add_executable_and_test(Example_VectorSlice_copy SOURCES
                                 Example_VectorSlice_copy.cpp)

morpheus_add_executable_and_test(Examples_DiaConvert SOURCES
                                 Examples_DiaConvert.cpp)

morpheus_add_executable_and_test(Examples_DiaMatrix SOURCES
                                 Examples_DiaMatrix.cpp)

morpheus_add_executable_and_test(Examples_DynamicResize SOURCES
                                 Examples_DynamicResize.cpp)

morpheus_add_executable_and_test(Examples_Init SOURCES Examples_Init.cpp)

morpheus_add_executable_and_test(Examples_Kokkos SOURCES Examples_Kokkos.cpp)

morpheus_add_executable_and_test(Examples_MatrixOperations SOURCES
                                 Examples_MatrixOperations.cpp)

morpheus_add_executable_and_test(Examples_Multiply SOURCES
                                 Examples_Multiply.cpp)

morpheus_add_executable_and_test(Examples_Unmanaged SOURCES
                                 Examples_Unmanaged.cpp)

morpheus_add_executable_and_test(Examples_Version SOURCES Examples_Version.cpp)
