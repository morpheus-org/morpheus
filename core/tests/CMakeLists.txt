message(STATUS "Core Tests are enabled")

morpheus_include_directories(${CMAKE_CURRENT_BINARY_DIR})
morpheus_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/category_files)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/format_files)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/test_utils)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/generic)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/DenseVector)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/DenseMatrix)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/Coo)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/Csr)
morpheus_include_directories(${Morpheus_SOURCE_DIR}/core/tests/Dia)

foreach(Tag Serial;OpenMP;Cuda;HIP)
  string(TOUPPER ${Tag} DEVICE)
  global_set(${Tag}_SOURCES)
endforeach()

add_subdirectory(generic)
add_subdirectory(DenseVector)
add_subdirectory(DenseMatrix)
add_subdirectory(Coo)
add_subdirectory(Csr)
add_subdirectory(Dia)

set(ALL_SOURCES)

if(MORPHEUS_ENABLE_SERIAL)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_Serial SOURCES TestMain.cpp
                                     ${Serial_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${Serial_SOURCES})
endif()

if(MORPHEUS_ENABLE_OPENMP)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_OpenMP SOURCES TestMain.cpp
                                     ${OpenMP_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${OpenMP_SOURCES})
endif()

if(MORPHEUS_ENABLE_CUDA)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_Cuda SOURCES TestMain.cpp
                                     ${Cuda_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${Cuda_SOURCES})
endif()

if(MORPHEUS_ENABLE_HIP)
  if(Morpheus_ENABLE_INDIVIDUAL_TESTS)
    morpheus_add_executable_and_test(UnitTest_HIP SOURCES TestMain.cpp
                                     ${HIP_SOURCES})
  endif()
  list(APPEND ALL_SOURCES ${HIP_SOURCES})
endif()

if(NOT Morpheus_ENABLE_INDIVIDUAL_TESTS)
  morpheus_add_executable_and_test(UnitTest SOURCES TestMain.cpp ${ALL_SOURCES})
endif()
