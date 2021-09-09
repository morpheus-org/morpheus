message(STATUS "Core Tests are enabled")

#
# Add test-only library for gtest to be reused by all the subpackages
#

set(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/gtest)

# need here for tribits
morpheus_include_directories(${GTEST_SOURCE_DIR})
morpheus_add_test_library(
  morpheus_gtest HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h SOURCES
  ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)

# avoid deprecation warnings from MSVC
target_compile_definitions(morpheus_gtest PUBLIC GTEST_HAS_TR1_TUPLE=0
                                                 GTEST_HAS_PTHREAD=0)

target_include_directories(morpheus_gtest PUBLIC ${GTEST_SOURCE_DIR})
if((NOT (MORPHEUS_ENABLE_CUDA AND WIN32)) AND (NOT ("${KOKKOS_CXX_COMPILER_ID}"
                                                    STREQUAL "Fujitsu")))
  target_compile_features(morpheus_gtest PUBLIC cxx_std_14)
endif()

# Suppress clang-tidy diagnostics on code that we do not have control over
if(CMAKE_CXX_CLANG_TIDY)
  set_target_properties(morpheus_gtest PROPERTIES CXX_CLANG_TIDY "")
endif()

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
  morpheus_add_executable_and_test(UnitTest_Serial SOURCES TestMain.cpp
                                   ${Serial_SOURCES})
  list(APPEND ALL_SOURCES ${Serial_SOURCES})
endif()

if(MORPHEUS_ENABLE_OPENMP)
  morpheus_add_executable_and_test(UnitTest_OpenMP SOURCES TestMain.cpp
                                   ${OpenMP_SOURCES})
  list(APPEND ALL_SOURCES ${OpenMP_SOURCES})
endif()

if(MORPHEUS_ENABLE_CUDA)
  morpheus_add_executable_and_test(UnitTest_Cuda SOURCES TestMain.cpp
                                   ${Cuda_SOURCES})
  list(APPEND ALL_SOURCES ${Cuda_SOURCES})
endif()

if(MORPHEUS_ENABLE_HIP)
  morpheus_add_executable_and_test(UnitTest_HIP SOURCES TestMain.cpp
                                   ${HIP_SOURCES})
  list(APPEND ALL_SOURCES ${HIP_SOURCES})
endif()

morpheus_add_executable_and_test(UnitTest SOURCES TestMain.cpp ${ALL_SOURCES})
